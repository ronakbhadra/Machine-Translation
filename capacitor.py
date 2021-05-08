import math
import cmath

def series_cap(c1,c2):
  if (c1==0 or c2==0):
    return 0
  if(1/c1+1/c2==0):
    return math.inf
  return 1.0/(1/c1+1/c2)

faulty_phases=[]
phases=['R1','Y1','B1','R2','Y2','B2']
correct_phases=[]

C=0.00001472

val=input("How many phases have faults?")

if(val!=0):
  print("Which phases have problems?(R1,Y1,B1,R2,Y2,B2)")

for i in range(int(val)):
  faulty_phases.append(input())

tot={}
for i in range(len(faulty_phases)):
    print("Enter the states of capacitors for phase "+faulty_phases[i])
    globals()[faulty_phases[i]]=[]
    for j in range(8):
          val=input()
          if(val==str(0)):
              globals()[faulty_phases[i]].append(C)
          elif(val==str(1)):
              globals()[faulty_phases[i]].append(math.inf)
          else:
              globals()[faulty_phases[i]].append(0)
    
    tot[faulty_phases[i]]=0
    for j in range(4):
       tot[faulty_phases[i]]=tot[faulty_phases[i]]+series_cap(globals()[faulty_phases[i]][2*j],globals()[faulty_phases[i]][2*j+1])
    
for ele in phases: 
    if ele not in faulty_phases:
        correct_phases.append(ele)  

for i in range(len(correct_phases)):
    globals()[correct_phases[i]]=[]
    for j in range(8):
       globals()[correct_phases[i]].append(C)
         
    
    tot[correct_phases[i]]=0
    for j in range(4):
       tot[correct_phases[i]]=tot[correct_phases[i]]+series_cap(globals()[correct_phases[i]][2*j],globals()[correct_phases[i]][2*j+1])


#print("Which phase capacitance do u want to know?")

print(tot)  

impedence={}
omega=314
for cap in tot:
  if(tot[cap]!=0):
     impedence[cap]=1.0/(omega*tot[cap])-0.145
  else:
     impedence[cap]=math.inf
print(impedence)

impedence_final={}

impedence_final['R']=series_cap(impedence['R1'],impedence['R2'])
impedence_final['Y']=series_cap(impedence['Y1'],impedence['Y2'])
impedence_final['B']=series_cap(impedence['B1'],impedence['B2'])

print(impedence_final)


V=33000.0/math.sqrt(3)

V_net = cmath.rect(V/impedence_final['R'],0)+cmath.rect(V/impedence_final['Y'],2*math.pi/3)+cmath.rect(V/impedence_final['B'],(-2)*math.pi/3)

print(V_net)

Z_net = 1.0/impedence_final['R']+1.0/impedence_final['Y']+1.0/impedence_final['B']

V_neutral = V_net/Z_net

print(V_neutral)
