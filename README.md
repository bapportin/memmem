Memory mapped Memory

for large language models (llm's) the context size (the number of tokens the llm can look into the passt) is the main cost for training and 
the main limitation. We test a way to make this context infinite by using an old idea 'memory mapped io'.

We think of the current position as the anchor and place the mapped memories at a fixed negative offset. For ease of understanding
we place the current position at 4096 and the memory map at 0. This way we can store the unrotated keys and query them using a kind
of LSH. The algorithm is an aproximation and the model ist not trained for that explicitly. We'll see if that actually works.
