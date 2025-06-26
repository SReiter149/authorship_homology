
skeleton = []
with open('../data/sloths/small_sloths_complete_complex.txt') as f:
    for line in f:
        skeleton.append(line)
    
print(skeleton)