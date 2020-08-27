from config import Config
from minion import (minion, ExpertMinion,
                    AllTheSingleLabelsMinion, RandomMinion)

def main():
  config = Config()
  names = config.names
  try:
    m = minion(0, names[0])
  except NotImplementedError:
    pass

  m = ExpertMinion(1, names[1])
  assert m.classify(1, 0)[1] == 0

  m = AllTheSingleLabelsMinion(2, names[2], 0)
  assert m.classify(1)[1] == 0

  m = AllTheSingleLabelsMinion(3, names[3], 1)
  assert m.classify(1)[1] == 1

  m = RandomMinion(4, names[4], config.labels)
  # this will fail 1 in 2^100 times
  x = []
  for i in range(100):
    x.append(m.classify(1)[1])
  x = set(x)
  assert 0 in x
  assert 1 in x

if __name__ == '__main__':
  main()
