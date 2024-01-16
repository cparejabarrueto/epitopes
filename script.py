#### python3 script.py
### .txt epitopes detail one by row
print('Input epitopes sequences')
txt1=input()
close=True
count1=0
count2=0
index=0
temp=""
array=[]
f = open(txt1, "r")
cadena=f.readlines()
f.close()

while close:
	for i in cadena:
		count1=len(i)
		
		for j in range(len(i)):
			if(i[j]!='\n'):
				if(j>=index):
					temp=temp+i[j]
					count2=count2+1
				if(count2==14):
					array.append(temp)
					count2=0
					temp=''
	index=index+1
	count2=0
	temp=''		
	if(index==count1):
		close=False
name=f'output_{txt1}'

print(name)
f = open(name, "w")
for i in array:
	f.write(i+'\n')
f.close()		
		
f = open(name, "r")
cadena=f.readlines()
f.close()	
new=[]
for i in cadena:
	insert=True
	for j in new:
		if(i==j):
			insert=False
	if(insert):
		new.append(i)

f = open(name, "a")
for i in new:
	f.write(i)
f.close()		
		
