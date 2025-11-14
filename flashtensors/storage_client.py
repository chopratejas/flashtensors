import grpc
import time
from .utils import init_logger
from .proto import storage_pb2, storage_pb2_grpc
from .config import get_server_address

logger = init_logger(__name__)

class StorageResponse:
    def __init__(self, success: bool, error: str = None, data=None):
        self.success = success
        self.error = error
        self.data = data

class StorageClient:
    def __init__(self, server_address=None):
        if server_address is None:
            server_address = get_server_address()
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = storage_pb2_grpc.StorageStub(self.channel)
        self.checkpoints_in_gpu = {}

    def __del__(self):
        # TODO: cleanup
        pass

    def load_into_cpu(self, model_path):
        request = storage_pb2.LoadModelRequest(
            model_path=model_path,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
                return False
            else:
                logger.error(f"Error: {e}")
                return False
        else:
            return response

    def unload_from_cpu(self, model_path):
        request = storage_pb2.UnloadModelRequest(
            model_path=model_path,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.UnloadModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return False
        else:
            return response

    def load_into_gpu(
        self, model_path, replica_uuid, tensor_copy_chunks, cuda_memory_handles
    ):
        logger.debug(f"load_into_gpu: {model_path}, {replica_uuid}")

        gpu_chunk_map = {}
        for device_uuid, chunks in tensor_copy_chunks.items():
            gpu_chunk_map[device_uuid] = storage_pb2.MemCopyChunkList(
                chunks=[
                    storage_pb2.MemCopyChunk(
                        src_offset=chunk[0],
                        size=chunk[1],
                        dst_offset=chunk[2],
                        handle_idx=chunk[3],
                    )
                    for chunk in chunks
                ]
            )
        cuda_handle_map = {}
        for device_uuid, handles in cuda_memory_handles.items():
            cuda_handle_map[device_uuid] = storage_pb2.MemCopyHandleList(
                handles=[
                    storage_pb2.MemCopyHandle(
                        cuda_ipc_handle=handle_str,
                    )
                    for handle_str in handles
                ]
            )
        request = storage_pb2.LoadModelRequest(
            model_path=model_path,
            replica_uuid=replica_uuid,
            chunks=gpu_chunk_map,
            handles=cuda_handle_map,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
            else:
                logger.error(f"Error: {e}")
            return False
        else:
            logger.info(f"Model loaded: {model_path}, {replica_uuid}")
            return response

    def confirm_model_loaded(self, model_path, replica_uuid, timeout=120):
        logger.info(f"confirm_model_loaded: {model_path}, {replica_uuid}")
        request = storage_pb2.ConfirmModelRequest(
            model_path=model_path,
            replica_uuid=replica_uuid,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        # Retry logic for server crashes/restarts
        max_retries = 5  # More retries for server restarts
        retry_delay = 10  # Longer delay to allow server to restart
        for attempt in range(max_retries):
            try:
                _ = self.stub.ConfirmModel(request, timeout=timeout)
                logger.info("Model loaded")
                return True
            except grpc.RpcError as e:
                logger.error(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
                if e.code() == grpc.StatusCode.CANCELLED:
                    logger.error("Model not loaded - cancelled")
                    return False
                elif e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.error("Model not loaded - service unavailable (possible crash)")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds (waiting for server to restart)...")
                        time.sleep(retry_delay)
                        # Wait for server to be available again
                        from .config import is_server_running
                        wait_count = 0
                        max_wait = 30  # Wait up to 30 seconds for server to restart
                        while wait_count < max_wait and not is_server_running():
                            time.sleep(2)
                            wait_count += 2
                            if wait_count % 10 == 0:
                                logger.info(f"   Still waiting for server to restart... ({wait_count}s)")
                        
                        if not is_server_running():
                            logger.warning("   Server did not restart in time, but continuing anyway...")
                        
                        # Reconnect to server
                        try:
                            self.channel.close()
                            self.channel = grpc.insecure_channel(self.server_address)
                            self.stub = storage_pb2_grpc.StorageStub(self.channel)
                            logger.info("   Reconnected to server")
                        except Exception as reconnect_error:
                            logger.warning(f"Failed to reconnect: {reconnect_error}")
                        continue
                    return False
                elif e.code() == grpc.StatusCode.INTERNAL:
                    logger.error("Model not loaded - internal error")
                    return False
                else:
                    logger.error(f"Model not loaded - status: {e.code()}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error in confirm_model_loaded: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return False
        return False

    def register_model(self, model_path) -> StorageResponse:
        logger.info(f"register_model: {model_path}")
        request = storage_pb2.RegisterModelRequest(model_path=model_path)
        try:
            response = self.stub.RegisterModel(request)
            logger.info("Model registered")
            return StorageResponse(success=True, data={"model_size": response.model_size})
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return StorageResponse(success=False, error=str(e))

    def get_server_config(self, max_retries=15, retry_delay=2.0):
        """Get server config with retry logic to handle startup race conditions."""
        request = storage_pb2.GetServerConfigRequest()
        
        for attempt in range(max_retries):
            try:
                response = self.stub.GetServerConfig(request)
                logger.info(f"Successfully connected to storage server on attempt {attempt + 1}")
                return {
                    "chunk_size": response.chunk_size,
                    "mem_pool_size": response.mem_pool_size,
                }
            except grpc.RpcError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to connect to storage server (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to storage server after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def clear_gpu_memory(self) -> StorageResponse:
        """Clear all GPU memory managed by the storage service."""
        logger.info("🧹 Clearing GPU memory via storage service...")
        request = storage_pb2.ClearMemRequest()
        try:
            response = self.stub.ClearMem(request)
            logger.info("✅ GPU memory cleared successfully via storage service")
            return StorageResponse(success=True)
        except grpc.RpcError as e:
            logger.error(f"❌ Failed to clear GPU memory via storage service: {e}")
            return StorageResponse(success=False, error=str(e))
