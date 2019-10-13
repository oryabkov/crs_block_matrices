There is a problem with free_vector and stop_use_vector in VectorOperations concept.
Whether these methods can throw an exceptions or not. If they can there is no 
guarantee that for every init there will be free call (the same for start/stop),
and we must add try catch somewhere into vector_wrap.free/stop_use. On the other
hand, if we block these exceptions what is the point of allowing them in the first 
place?