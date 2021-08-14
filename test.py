
def ans(s):
    a = [] 
    n = len(s)

    k = 0 
    if n==1:
            return s[0]
    for i in range(1,n):
        
        
        if (s[i]=="+" or s[i]=="-"  or s[i]=="/" or s[i]=="*") and (s[i-1] != "+" and s[i-1] != "-" and s[i-1] != "/" and s[i-1] != "*"):     
            if(s[i]=="+"):
                k = int(s[i-2])+int(s[i-1])
            elif(s[i]=="-"):
                k = int(s[i-2])-int(s[i-1])
            elif(s[i]=="/"):
                k = int(s[i-2])/int(s[i-1])
            elif(s[i]=="*"):
                k = int(s[i-2])*int(s[i-1])
            a.append(k)
        elif (s[i]=="+" or s[i]=="-"  or s[i]=="/" or s[i]=="*") and (s[i-1] == "+" or s[i-1] == "-" or s[i-1] == "/" or s[i-1] == "*"): 
            a.append(s[i])
    print(ans(a))

s = input().split(" ")
ans(s)




