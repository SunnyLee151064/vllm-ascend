import numpy as np
num_decode_pcp_size = 3
max_scheduled_tokens = 5
num_decode_tokens = 2
res = np.repeat(np.arange(num_decode_pcp_size) * max_scheduled_tokens, num_decode_tokens)

print(res)
