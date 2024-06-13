# utils 

def mems():
    cmd = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
    return [int(x) for x in cmd.decode('utf-8').strip().split('\n')]

def identity(x):
    return x

def argmax(xs, *, key=identity):
    return max(enumerate(xs), key=lambda x: key(x[1]))[0]

def max_mem():
  return argmax(mems())

def dev_t(x, t=int):
  match t:
    case bis.int:
      return x 
    case bis.str:
      return f'cuda:{x}'
    case torch.device:
      return torch.device(f'cuda:{x}')
    case _:
      raise ValueError(f'dev_t: {t}')
     
def set_seed(seed):
  print(f"warning: setting seed {seed}")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False