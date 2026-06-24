class MidlinesChunkObj(): #made of a bunch of midlines

    def __init__(self, start_frame,chunk_start_i, chunk_end_i, midlines_chunk, inner_worm_points_chunk,bad_frames, ):
    	self.start_frame = start_frame
    	self.chunk_start_i =  chunk_start_i
    	self.chunk_end_i = chunk_end_i
    	self.midlines_chunk = midlines_chunk
    	self.bad_frames = bad_frames
    	self.inner_worm_points_chunk = inner_worm_points_chunk