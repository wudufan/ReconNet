extern "C" int cSetDevice(int device)
{
	return cudaSetDevice(device);
}
