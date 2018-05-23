import time
import math

def local_clock(shift=-25200):
	return time.asctime(time.localtime(time.time() + shift))

def chunk(l, chunks):
	chunk_list = []
	chunk_size = math.floor(len(l) / chunks)
	for i in range(chunks - 1):
		chunk_list.append(l[chunk_size*i:chunk_size*(i+1)])
	chunk_list.append(l[chunk_size*(chunks - 1):])
	return chunk_list