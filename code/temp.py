with open('../data/math/math.json', 'r') as f:
    text = f.read()
    replaced_text = text.replace('}{', ', ')

with open('../data/math/math2.json', 'w') as f:
    f.write(replaced_text)
