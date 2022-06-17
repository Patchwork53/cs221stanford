from collections import defaultdict

a = defaultdict(float)

a['east']=1
a['north']=-1

for x in a:
    print(x)