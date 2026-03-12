def read(FileName):
    samples = [] 
    with open(FileName,'r') as f:
        for i,sline in enumerate(f.readlines()[1:]):
            line = sline.split()
            samples.append(line) 
    return samples
