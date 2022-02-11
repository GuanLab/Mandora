import numpy as np
import sys
  
individual_all=[]
f=open('individual_all.txt')
for line in f:
    line=line.rstrip()
    individual_all.append(line)

f.close()
individual_all=np.array(individual_all)

print(sys.argv)
np.random.seed(int(sys.argv[1]))
np.random.shuffle(individual_all)
ratio=[0.5,0.3,0.2]

num1 = int(len(individual_all)*ratio[0])
individual_unet = individual_all[:num1]
num2 = num1 + int(len(individual_all)*ratio[1])
individual_lgbm = individual_all[num1:num2]
individual_test = individual_all[num2:]

individual_unet.sort()
individual_lgbm.sort()
individual_test.sort()

np.savetxt('individual_unet.txt', individual_unet, fmt='%s')
np.savetxt('individual_lgbm.txt', individual_lgbm, fmt='%s')
np.savetxt('individual_test.txt', individual_test, fmt='%s')


