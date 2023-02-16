#directory = "/Users/hosseindehghani/Desktop/Hafezi/Codes/"
#cd(directory)
# module JuliaCliffordMai
#using Distributed
#addprocs(1)
# we use QCliffordTimeEvolveAncilla or QCliffAncilla. 
# The latter does not work when we create trajectories for learning. Learning fails with the produced trajectories.


using Base.Threads
using SharedArrays
using Pkg
using Plots

#Pkg.add("NPZ")
#Pkg.add("JLD")tran
#Pkg.add("JLD2")
#Pkg.add("StatsBase")
#Pkg.add("VegaLite")
#Pkg.add("VegaDatasets")
#Pkg.add("Distributed")
#Pkg.add("DataFrames")
#Pkg.add("FileIO")
#Pkg.add("Plots")
#Pkg.add("QuantumClifford")
#Pkg.add("CSV")
using FileIO
using NPZ
#using JLD
using StatsBase
#using VegaLite, 
using VegaDatasets
using Distributed
using CSV, DataFrames, FileIO
using QuantumClifford
#Pkg.add("JLD")
#addprocs(4)

using JLD2
using QuantumClifford
#Pkg.add("JLD")

using JLD2

file = jldopen("ru2.jld2", "r")
ru2 = file["ru2"]
close(file)

function returnPaul(p,sign=0x0)
    n=length(p)
    return PauliOperator(sign,convert.(Bool,p[1:n÷2]),convert.(Bool,p[n÷2+1:n]))
end

function RandUnitaries()
    n = 0
    Nline = 2880

    unitaries = zeros(2880, 4)
    x = [0:1:10;]
    #println(x)
    
    #println(x[1:2:10])

    open("twoUnitaries.txt") do f 
    
    
  
      # line_number 
        line = 0   
  
      # read till end of file 
        while ! eof(f) #&& line<10  
  
         # read a new / next line for every iteration            
            s = readline(f)           
            line += 1
            #println("$line . $s") 
            #println(s[1:2:7])  
            #println("Int(x[1])", Int(x[1]))
            unitaries[line, 1] = parse(Int64, s[1])
            unitaries[line, 2] = parse(Int64, s[3])
            unitaries[line, 3] = parse(Int64, s[5])
            unitaries[line, 4] = parse(Int64, s[7])
            #println("unitaries[line, :] = ", unitaries[line, :])
            #print(typeof(parse(Int64, s[1:2:7])))
    #        unitaries[line, 4] = parse(Int64, s[1])
            
    #        println(s[1])
    #        println(s[:])
            #print(s[1])
            #println("line = ", line)
            #println("unitaries[line, 1:4]", unitaries[line, 1:4])        
      end 
    end 
    #println(unitaries[1:10, :])

    #for x in f
    #    if n<Nline
            #println(len(x))
            #println(n, 'n\n', x[2])
            #println('x3', x[0:8:2], '\n')
    #        unitaries[n, 0] = int(x[0])
    #        unitaries[n, 1] = int(x[2])
    #        unitaries[n, 2] = int(x[4])
    #        unitaries[n, 3] = int(x[6])
            #println('unitaries', unitaries[n, :])
    #        n += 1
    #    end
    #end

    #println("final n", n)
    #println(unitaries[1, :])
    randU = zeros(Nline, 4)
    n=0
    #println("unitaries[n+1:n+4, 2]", size(unitaries[n+1:n+4, 2]))
    #println("randU[n+1, 0:4]", size(randU[n+1, 1:4]))
    #println(size(unitaries))
    #println("unitaries = ", unitaries[1, 1])


    for n in 0:Int(Nline/4)-1
        #println('n', n)
        randU[4*n+1, 1:4] = copy(unitaries[4*n+1:4*n+4, 3]) # Z1 line: 0th line of n'th randU
        randU[4*n+2, 1:4] = copy(unitaries[4*n+1:4*n+4, 1])# X1 line: 1st line of n'th randU
        randU[4*n+4, 1:4] = copy(unitaries[4*n+1:4*n+4, 4]) # Z2 line: 3rd line of n'th randU
        randU[4*n+3, 1:4] = copy(unitaries[4*n+1:4*n+4, 2]) # X2 line: 2nd line of n'th randU    
    end

    #println("randU[0:4, :] = ", randU[1:4, :])
    swap = zeros(Nline)
    swap[1:Nline] = randU[:, 1]
    randU[:, 1] = randU[:, 4]
    randU[1:Nline, 4] = swap[1:Nline]   

    #print(randU[1000:1004, :])
    #println("randU \n", randU[2000:2008, :])
    #print('unitaries\n', unitaries[0:4, :])
    return randU
end


function initProdState(N, ancilla)
    N = Int(N)
    if !Bool(ancilla)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are Z operators
            Ginit[i+N, 2*i] = 1 # X stabilizers are X operators
        end

    else
        N = Int(N)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are X operators
            Ginit[i+N, 2*i] = 1 # X stabilizers are Z operators
        end
    end    
        #print('Sinit Shape \n', Sinit.shape)
    return Ginit 
end


function initProdStateShort(N, ancilla)
    # We remove the sign bit at the end of the Pauli operators. 
    
    N = Int(N)
    if !Bool(ancilla)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are Z operators
        end
    else
        N = Int(N)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are X operators
        end
    end    
    return Ginit 
end



function fourbitFunc(a, b, c, d)
    if a==0 && b==0
        return 0
    end
    if a==1 && b==1
        return c-d
    end
    if a==1 && b==0
        return d*(1-2*c)
    end
    if a==0 && b==1
        return c*(2*d-1)
    end    
end


function xor(v1, v2)
    return [Int(sum(x)%2) for x in zip(v1, v2)]
end

function PauliProd(P1, P2)
    # Takes two vectorial Pauli Operators. 
    #println("length(P1) = ", length(P1))
    N = Int(round((length(P1)-1)/2)) 
    #println("N = ", N)
    #print('N', N)
    #print('P1[:int(2*N)]', P1[:int(2*N)])
    #print('P1 ',P1)
    #print('P2 ',P2)    
    r1 = P1[2*N+1]
    r2 = P2[2*N+1]
    #println("r1 r2", r1, r2)
    sumG = 0
    #print('len P1', shape(P1))
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)])
    #print("P1 P2 sumPauli = ", P1, P2, sumPauli)
    for i in 1:N
        sumG += fourbitFunc(P1[2*i-1],P1[2*i],P2[2*i-1],P2[2*i])
    end
    #println("end of sumPauli = ", sumPauli)
    #println("((2*r1 + 2*r2 + sumG) % 4)/2 = ", ((2*r1 + 2*r2 + sumG) % 4)/2)
    return sumPauli, abs(((2*r1 + 2*r2 + sumG) % 4)/2)
end



function PauliProdShort(P1, P2)
    N = Int(round((length(P1)-1)/2))
    sumG = 0
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)])
    
    return sumPauli
end



function inputFunc(a, b, c=3)
    println("c = ", c)
    if isempty(c)
        println(a)
        println("c = isempty")
    end
end



function sumx1x2(x1, x2)
    return cx1, x1[1]+x2
end

function enlargeU_IZXY(U)
    
    N = Int(round(size(U)[1]/2))

    g = zeros(N, 2, 2, 2*N+1)
    for i in 1:Int(N)
        g[i, 2, 1, 1:2*N+1] = copy(U[2*i, 1:2*N+1])
        g[i, 1, 2, 1:2*N+1] = copy(U[2*i-1, 1:2*N+1])
        g[i, 2, 2, 1:2*N] = copy(xor(U[2*i-1, 1:2*N], U[2*i, 1:2*N]))
        prodUxUz, prodSignXSignZ = PauliProd(U[2*i-1, :], U[2*i, :])
        g[i, 2, 2, 2*N+1] = ((1 + 2*prodSignXSignZ) % 4)/2
    end
    return g
end


function PauliTransform(P, U)
    #println(size(U))
    N = Int(round(size(U)[1]/2))
    #println("cursor")
    numRows = size(P)[1]
    newP = zeros(size(P))
    for r in 1:numRows
        PBit = P[r, 2*N+1]
        #println(size(P[r, 1:2*N]))
        #println(size(U))
        #println("P", P[r, 1:2*N+1])
        transformP = transpose(P[r, 1:2*N])*U.%2
        #println("transformP = ", transformP)
        #transformP[1:2*N] = [i%2 for i in transformP[1:2*N]]
        #println(transformP)
        enlargeU = enlargeU_IZXY(U)
        #println("enlargeU = ", enlargeU)
        SBitProd = 0

        sumPauli = zeros(1, 2*N+1)
        newP[r, 1:2*N] = transformP[1:2*N]
        #P[r, 2*N] = P[r, 2*N]

        SBitProd = 0

        #print('before loop')

        for n in 1:N-1
            #print('n =', n)
            #println("P[r, 2*n]", P[r, 2*n])
            #print("Pauli from PauliTransform")
            #println("(enlargeU) = ", enlargeU[n, Int(P[r, n])+1, Int(P[r, n+1])+1, 1:2*N+1])
            #println("sumPauli = ", sumPauli)
            tempSumPauli = PauliProd(enlargeU[n, Int(P[r, n])+1, Int(P[r, n+1])+1, 1:2*N+1], sumPauli)
            #println("tempSumPauli = ", tempSumPauli)
            SBitProd = tempSumPauli[2]
            sumPauli[1, 1:2*N] = tempSumPauli[1][1:2*N]
            sumPauli[1, 2*N+1] = tempSumPauli[2]
            #println("after loop")
            #println("sumPauli = ", sumPauli)
            #print('SBitProd', SBitProd)
        end
        transformPBit = (PBit + SBitProd)%2
        #transformPBit = transformPBit % 2
        
        newP[r, 1:2*N] = copy(transformP[1:2*N])
        newP[r, 2*N+1] = copy(transformPBit)
        #print('newP[r, :]', newP[r, :])
    end    
    
    return newP
end



function PauliTransformShort(P, U)
    #println(size(U))
    N = Int(round(size(U)[1]/2))
    #println("cursor")
    numRows = size(P)[1]
    newP = zeros(size(P))
    for r in 1:numRows        
        transformP = transpose(P[r, 1:2*N])*U.%2
        newP[r, 1:2*N] = copy(transformP[1:2*N])
    end    
    return newP
    
end



function symplecticProd(P1, P2)
    N = Int((length(P1)-1)/2)
    sumP = 0
    for i in 1:N
        #print('P1[2*i]*P2[2*i+1] + P1[2*i+1]*P2[2*i]', P1[2*i]*P2[2*i+1] + P1[2*i+1]*P2[2*i])
        #print('2i, 2i+1', 2*i, 2*i+1)
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
        #print(sumP)
    end
    return sumP%2
end



function symplecticProdShort(P1, P2)
    # The sign bit is removed.
    N = Int((length(P1)-1)/2)
    sumP = 0
    for i in 1:N
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
    end
    return sumP%2
end


function measurePauli(g, S)
    #println("g = ", g)
    #println("size(g)[2] = ", size(g)[2])
    N = Int((size(g)[2]-1)/2)
    #println("N ", N)
    orthoSymProd = zeros(N)
    #println("size g")
    #println(size(g))
    ZantiCommute = zeros(size(g))
    #println("ZantiCommute")
    #println(ZantiCommute)
    XantiCommute = zeros(size(g))   
    Zindex = Int64[]
    Xindex = Int64[]
    Zcounter = 0
    Xcounter = 0
    
    newSZindex1 = zeros(size(g))
    newS = zeros(Float64, size(S))
    #newS[1, 1] = 1.4
    newStabX = zeros(1, 2*N+1)
    newStabZ = zeros(1, 2*N+1)
    
    measureRes = NaN  # "measureRes = x" means measure_Result = i^{2x}
    deterministic = 0
    #println("g, and S[:, :]")
    #printMatrix(g)
    #printMatrix(S)
    
    for i in 1:N
        
        #println("size(g, S[i,:]) = ", size(g), size(S[i,:]))
        
        Zprod = symplecticProd(g, S[i,:])
        if Zprod == 1
            Zcounter += 1
            #println("Zcounter = ", Zcounter)
            #println("Zindex = ", Zindex)
            
            append!(Zindex, Int(i))
            if Zcounter == 1
                #println("ZantiCommute \n", size(ZantiCommute))
                #print('S[i,:]', size(S[i,:]))
                #println("S[i,:] \n")
                #println(size(S[i,:]))

                #println("size(ZantiCommute) before", size(ZantiCommute))
                ZantiCommute[1, :] = copy(S[i,:])
                #println("size(ZantiCommute) after", size(ZantiCommute))                
                #println("ZantiCommute")                
                #println(ZantiCommute)                
                
            else
                #println("else")
                #ZantiCommute = [ZantiCommute, S[i,:]]
                #println(ZantiCommute)
                #println("size ZantiCommute", size(ZantiCommute))
                #println("size S[i,:]", size(S[i,:]))
                ZantiCommute = vcat(ZantiCommute, S[i,:]')
                #println("after vcat")
                #println("ZantiCommute")
                #println(ZantiCommute)
                #ZantiCommute = append!(ZantiCommute, [S[i,:]], axis=0)
            end
        end
        #println("after Zprod == 1, ZantiCommute = ", ZantiCommute)
        Xprod = symplecticProd(g, S[i+N,:])
        if Xprod == 1    
            Xcounter += 1
            #println("Xindex = ", Xindex)
            
            append!(Xindex, Int(i+N))
            #println("Xindex after = ", Xindex)
            
            if Xcounter == 1
                #print('XantiCommute', size(XantiCommute))
                #print('S[i+N,:]', size(S[i+N,:]))
                #println("before Xanticommute")
                #println("size(XantiCommute[1, :])", size(XantiCommute[1, :]))
                #println("XantiCommute = ", XantiCommute)
                #println("S[i+N,:]", S[i+N,:])
                
                #println("size(S[i+N,:])", size(S[i+N,:]))
                XantiCommute[1, :] = copy(S[i+N,:])
                #println("after Xanticommute")
            else
                #println("XantiCommute before append")
                XantiCommute = vcat(XantiCommute, S[i+N,:]')                
                #XantiCommute = append!(XantiCommute, [S[i+N,:]], axis=0)
                
            end
        end
    end
    #newS[1, 1] = 1.8
    newS = copy(S)
    #println("Zcounter = ", Zcounter)
    #println("after copy newS")
    if Zcounter == 0
        #println("S in Zcounter==0")
        
        deterministic = 1
        #println("deterministic = 1")
        #exit()
        plusMinusG = zeros(Int(2*N)+1)
        #println('g \n', g)
        
        for i in 1:N
            coefX = Int(symplecticProd(g, S[i+N, :]))
            #println("coefX = ", coefX)
            
            tempPlusMinusGProd = PauliProd(coefX*S[i, :], plusMinusG)
            tempPlusMinusG = xor(coefX*S[i, 1:2*N], plusMinusG)

            plusMinusG[1:2*N] = copy(tempPlusMinusG)
            plusMinusG[2*N+1] = copy(tempPlusMinusGProd[2])
            #println("plusMinusG = ", plusMinusG)  
        end
        for i in 1:2*N 
            #println("plusMinusG[i] = ", plusMinusG[i])
            #println("g[i] = ", g[i])
            if plusMinusG[i] != g[i]
                #println("not equal")
                #exit()
            end
        end
        #println("plusMinusG = \n")
        #println(plusMinusG)
        #println("g = \n")
        #println(g)
        if plusMinusG[2*N+1] == g[2*N+1]
            measureRes = 0
            #print("equal")
        else
            #print("else equal")
            measureRes = (g[2*N+1]-plusMinusG[2*N+1])%2
        end    
        if measureRes < 0
            measureRes += 2
        end
        #if plusMinusG[2*N]==1
        #    measureRes = -1
        #end
        #println("Zcounter == 0 and measureRes = -1")
        #else:
            #println('Zcounter == 0 and measureRes = 1')
            
        for n in 1:N
            orthoSymProd[n] = symplecticProd(newS[n, :], newS[n+Int(N), :])
        end
        #println("orthoSymProd =", orthoSymProd)
    elseif Zcounter >= 1
        #println("S in Zcounter>1 before ", S)
        #println("g = ", g)
        
        #coin = random.choice([-1, 1], 1, [.5, .5])[0]
        
        #println('Zindex', Zindex)
        #println('Xindex', Xindex)        

        for i in 2:Zcounter
            #println("i in 2:Zcounter")
            #println("S[Zindex[1] = ", S[Zindex[1]])
            #println("S[Zindex[i], :] = ", S[Zindex[i], :])
            
            tempProdZ = PauliProd(S[Zindex[i], :], S[Zindex[1], :])
            #println("size(tempProdZ[1]) = ", size(tempProdZ[1]))
            #println("size(tempProdZ[2]) = ", size(tempProdZ[2]))            
            #println("(tempProdZ[1]) = ", tempProdZ[1])
            #println("(tempProdZ[2]) = ", tempProdZ[2])
            
            newStabZ[1, 1:2*N] = copy(tempProdZ[1][1:2*N]) 
            newStabZ[1, 2*N+1] = copy(tempProdZ[2])
            #println("tempProdZ = ", tempProdZ)
            #println("newStabZ = ", newStabZ)
            newS[Zindex[i], :] = copy(newStabZ[:])
            #println("newS[Zindex[i], :] = ", newS[Zindex[i], :])
        end
        
        
        for i in 1:Xcounter
            #println("Zindex = ", Zindex)
            #println("Xindex = ", Xindex)
            #println("Zindex[1] = ", Zindex[1])
            #println("Xindex[i] = ", Xindex[i])   
            #println("PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :])", PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :]))
            
            tempProdX = PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :])
            #println("size(tempProdX) = ", size(tempProdX))
            #println("tempProdX[1] ", tempProdX[1])
            #println("tempProdX[2] ", tempProdX[2]) 

            newStabX[1, 1:2*N] = copy(tempProdX[1][1:2*N])
            newStabX[1, 2*N+1] = copy(tempProdX[2])
            
            #tempProdX, tempSignX = PauliProd(S[Xindex[i], :], S[Zindex[1], :])            
            #newStabX = append!(tempProdX, tempSignX)            
            #println("tempProdX = ", tempProdX)
            #println("newStabX = ", newStabX) 
            #println("size(newStabX[1, :])", size(newStabX[1, :]))
            #println("Xindex[i] = ", Xindex[i])
            #println("newStabX[1, :] = ", newStabX[1, :])
            #println("size(newS[Xindex[i], :])", size(newS[Xindex[i], :]))

            newS[Xindex[i], 1:2*N+1] = copy(newStabX[1, 1:2*N+1])
            #newS[1, 9] = 1.3
            #println("after Xindex assignment")
            #tempProdX, tempSignX = PauliProd(S[Xindex[i], :], S[Zindex[0], :])            
            #newStabX = np.append(tempProdX, tempSignX)            
            #newS[Xindex[i], :] = np.copy(newStabX[:])
            
        end
        for m in 1:N
            #println("m in 1:N")
            
            orthoSymProd[m] = symplecticProd(newS[m, :], newS[m+Int(N), :])
        end

        #println("orthoSymProd = ", orthoSymProd)
        #println("Intermediate Zindex = ", Zindex)         
        newSZindex1[:] = copy(newS[Int(Zindex[1]), :])

        newS[Int(Zindex[1]), :] = copy(g)
        
        randVec = rand(Float64)        
        coin = [Bool(i<0.5) for i in randVec]   
        #println("coin = ", coin[1])
        
        if Bool(coin[1])
            newS[Int(Zindex[1]), Int(2*N)+1] = (g[Int(2*N)+1]+1)%2  #Int((g[Int(2*N)+1]+1)%2)  
            measureRes = 1 # measureRes = x means i^{2x}; x = 1 -> m=-1
        else
            measureRes = 0 # measureRes = x means i^{2x}; x=0 -> m=0
        end
        newS[Int(Zindex[1])+N, :] = copy(newSZindex1[:])            

    end
    #println('newS final \n', newS)   
    #println("Final Zindex = ", Zindex)
    tempMeasure = 0
    if measureRes == 0
        tempMeasure = 1 #S_z = -1
    elseif isnan(measureRes)
        tempMeasure = 0     # No measurement           
    elseif measureRes == 1.0
        tempMeasure = 2  # S_z = +1
    elseif measureRes == 1.5
        tempMeasure = 1.5  # S_z = +1
    elseif measureRes == 0.5
        tempMeasure = 0.5  # S_z = +1        
    end
    measureRes = tempMeasure
    
    return newS, measureRes, deterministic #ZantiCommute, XantiCommute
end





function measurePauliShort(g, S)
    N = Int((size(g)[2]-1)/2)

    orthoSymProd = zeros(N)
    ZantiCommute = zeros(size(g))
    XantiCommute = zeros(size(g))   
    Zindex = Int64[]
    Xindex = Int64[]
    Zcounter = 0
    Xcounter = 0
    
    newSZindex1 = zeros(size(g))
    newS = zeros(Float64, size(S))
    #newS[1, 1] = 1.4
    newStabX = zeros(1, 2*N+1)
    newStabZ = zeros(1, 2*N+1)
    
    measureRes = NaN  # "measureRes = x" means measure_Result = i^{2x}
    deterministic = 0
    
    for i in 1:N
        
        Zprod = symplecticProd(g, S[i,:])
        if Zprod == 1
            Zcounter += 1
            
            append!(Zindex, Int(i))
            if Zcounter == 1
                ZantiCommute[1, :] = copy(S[i,:])                
            else
                ZantiCommute = vcat(ZantiCommute, S[i,:]')
            end
        end
        
        Xprod = symplecticProd(g, S[i+N,:])
        if Xprod == 1
            Xcounter += 1
            append!(Xindex, Int(i+N))
            
            if Xcounter == 1
                XantiCommute[1, :] = copy(S[i+N,:])
            else
                XantiCommute = vcat(XantiCommute, S[i+N,:]')                                
            end
        end
    end
    newS = copy(S)
    if Zcounter == 0
        deterministic = 1
        plusMinusG = zeros(Int(2*N)+1)
        for i in 1:N
            coefX = Int(symplecticProd(g, S[i+N, :]))            
            tempPlusMinusGProd = PauliProd(coefX*S[i, :], plusMinusG)
            tempPlusMinusG = xor(coefX*S[i, 1:2*N], plusMinusG)
            plusMinusG[1:2*N] = copy(tempPlusMinusG)
            plusMinusG[2*N+1] = copy(tempPlusMinusGProd[2])
        end
        for i in 1:2*N 
            if plusMinusG[i] != g[i]
                #println("not equal")
                #exit()
            end
        end

    elseif Zcounter >= 1

        for i in 2:Zcounter
            tempProdZ = PauliProd(S[Zindex[i], :], S[Zindex[1], :])            
            newStabZ[1, 1:2*N] = copy(tempProdZ[1][1:2*N]) 
            newStabZ[1, 2*N+1] = copy(tempProdZ[2])
            newS[Zindex[i], :] = copy(newStabZ[:])
        end
        
        
        for i in 1:Xcounter
            tempProdX = PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :])
            newStabX[1, 1:2*N] = copy(tempProdX[1][1:2*N])
            newStabX[1, 2*N+1] = copy(tempProdX[2])
            
            newS[Xindex[i], 1:2*N+1] = copy(newStabX[1, 1:2*N+1])
            
        end
        for m in 1:N            
            orthoSymProd[m] = symplecticProd(newS[m, :], newS[m+Int(N), :])
        end

        newSZindex1[:] = copy(newS[Int(Zindex[1]), :])

        newS[Int(Zindex[1]), :] = copy(g)
        
        randVec = rand(Float64)        
        coin = [Bool(i<0.5) for i in randVec]   
        
        if Bool(coin[1])
            newS[Int(Zindex[1]), Int(2*N)+1] = (g[Int(2*N)+1]+1)%2  #Int((g[Int(2*N)+1]+1)%2)  
        end
        newS[Int(Zindex[1])+N, :] = copy(newSZindex1[:])            
    end
    return newS, deterministic    
end


function projectOut(Matrix, RegionA)
    # RegionA: The first element is the starting point of region A, 
    # and the second component is the length of region A
    # sizeA = size(RegA)[1]
    numRow = size(Matrix)[1]
    #println("numRow = ", numRow)
    #println("size(Matrix)", size(Matrix))
    #println("numRow = ", numRow)
    #println("regionA2 = ", RegionA[2])    
    projectA = zeros(Int(numRow), Int(RegionA[2]))
    #println("RegionA", RegionA)
    projectA[1:numRow, 1:RegionA[2]] = copy(Matrix[1:numRow, RegionA[1]:RegionA[1]+RegionA[2]-1])
    #println("projectA[1:numRow, 1:RegionA[2]]")
    return projectA
end

function clippingGauge(A)
    # Assumption: [M] = m * n
    m = size(A)[1]
    n = size(A)[2]
    h = 1
    k = 1
    tempArray = zeros(1, n)
    #println("m, n = ", m, n)
    
    while h <= m && k <= n
        #println("inside while")
        #println("A[h:m, k] = ", A[h:m, k])
        #println("findmax(A[h:m, k]) = ", findmax(A[h:m, k])[2])
        i_max = h-1+findmax(A[h:m, k])[2]
        #println("i_max = ", i_max)    
        #println("k = ", k)
        #println("A[i_max, k]", A[i_max, k])
        #argmax((A[h:m, k]), axis = None)
        if A[i_max, k] == 0
            k += 1
            #println("A[i_max, k] == 0")            
            continue            
        else
            #println("tempArray[1, :] = A[h, :]")            
            tempArray[1, :] = A[h, :]

            A[h, :] = A[i_max, :]
            A[i_max, :] = tempArray[1, :]
        end
        for i in h+1:m
            #println("for i in h+1:m]")            
            
            if A[h, k]==0
                #println("A[h, k]==0 \n", A)
            end
            f = A[i, k]/A[h, k]
            #print('f', f)
            A[i, k] = 0
            for j in k+1:n
                #print('i, j', i, j)
                A[i,j] = abs((A[i,j] - A[h,j]*f)%2)
            end
            #print('A \n', A)
        end
        h += 1
        k += 1
    end
    echelonA = zeros(m, n)
    #println("after while")
    for i in 1:m
        echelonA[i, :] = copy(A[m-i+1, :])
    end
    #println("echelonA = ", echelonA)
    

    
    zeroRow = 0
    for i in 1:size(echelonA)[1]
        for j in 1:size(echelonA)[2]
            #println('i, j', i, j)
            if (echelonA[i, j]==0) && j != size(echelonA)[2]
                continue
            elseif (echelonA[i, j]==0) && j == size(echelonA)[2]

                zeroRow += 1
                break
            else
                break 
            end
        end
    end
    rank = size(echelonA)[1] - zeroRow      
    
    #println("rank = ", Arank)        
    return rank #, echelonA
end

function Entropy(M, A)
    #println("inside Entropy 0")
    lengthA = A[2]
    #println("inside Entropy 0-1")
    projA = projectOut(M, A)
    
    #
    for i in 1:size(projA)[1]
        #println("projA[$i, :] =", projA[i, :])
    end
    for i in 1:size(M)[1]
        #println("M[$i, :] =", M[i, :])
    end
#    println("M = ")
#    printMatrix(M)

    #println("inside Entropy 1")
    
    #println("projA = ")
    #printMatrix(projA)
    
    rankProjA = clippingGauge(projA)
    #println("inside Entropy 2")
    
    #println("rank projected out = ", rankProjA[1])
    #println("lengthA = ", lengthA)
    SA = rankProjA - lengthA/2
    
    return SA
end

function checkOrtho(G)
    #print('G in checkOrtho\n', G)
    #print('size(G) in checkOrtho', size(G))
    
    N = Int(size(G)[1]/2)
    #print('N', N)
    ortho = zeros(2*N, 2*N)
    for i in 1:2*N
        for j in 1:2*N
        #print('G[i,  :2*N]', G[i, :])
        #print('G[i+N,:2*N]', G[i+N, :])        
            ortho[i, j] = symplecticProd(G[i, :], G[j, :])
        end
    end

    for i in 1:2*N
        #println("G[$i, :] = ", ortho[i, :])        
    end
    
    return ortho
end   


function convertStabRep(S)
    N = Int((size(S)[2]-1)/2)
    #println("N inside convertStabRep = ", N)
    signVec = zeros(UInt8, N)
    for i in 1:N
        signVec[i] = 0x0
    end
    Xbits = zeros(Bool, N, N)
    Zbits = zeros(Bool, N, N)    
    signI = 0x0
    for i in 1:N
        if S[i, 2*N+1]==0
            signI = 0x0            
        elseif S[i, 2*N+1]==0.5
            signI = 0x1
        elseif S[i, 2*N+1]==1
            signI = 0x2            
        elseif S[i, 2*N+1]==1.5
            signI = 0x3            
        end
        #println("signI = ", signI)
        signVec[i] = signI
        for j in 1:N
            Xbits[i, j] = Bool(S[i, 2+2*(j-1)])    
            Zbits[i, j] = Bool(S[i, 1+2*(j-1)])
        end
    end
    #println(Xbits)
    #println(Zbits)    
    #println("signVec in convertStabRep = ", signVec)
    stab = Stabilizer(signVec,Xbits,Zbits)
    #println("stab = ", stab)
    return stab
end

function convBackStab(S)
    S_XZRep = stab_to_gf2(S)
    #println(sXZRep)
    size1 = Int(size(S_XZRep)[1])
    size2 = size(S_XZRep)[2]
    #println(size(sXZRep))
    SPhase = S.phases
    myRep = zeros(size1, size2+1)
    #println(fieldnames(s))
    #println("Int(size2/2) = ", Int(size2/2))
    #println("Int(size1) = ", Int(size1))
    Xbits = zeros(Bool, size1, Int(size2/2))
    Zbits = zeros(Bool, size1, Int(size2/2))
    #println("Xbits = ", Xbits)
    #A = [5, 7, 6, 4, 1, 0]
    #println("A = ", A[2:2:6])
    #println("A = ", A[1:2:6])
    for i in 1:size1
        #println(1:Int(size2/2))
        #println(Int(size2/2)+1:size2)    
        Xbits[i, :] = S_XZRep[i, 1:Int(size2/2)]
        Zbits[i, :] = S_XZRep[i, Int(size2/2)+1:size2]    
        
        #println("Zbits[i, :] = ", Zbits[i, :])
        #println("Xbits[i, :] = ", Xbits[i, :])        
        myRep[i, 1:2:size2] = Zbits[i, :]
        myRep[i, 2:2:size2] = Xbits[i, :]        
        sign = 0
        if SPhase[i]==0x0
            sign = 0
        elseif SPhase[i]==0x1
            sign = 0.5
        elseif SPhase[i]==0x2            
            sign = 1
        elseif SPhase[i]==0x3
            sign = 1.5          
        end
        myRep[i, size2+1] = sign
    end   
    return myRep
    println("myRep = ", myRep)
end

function convertPauliRep(P)
    N = Int((size(P)[2]-1)/2)
    Xbits = zeros(Bool, N)
    Zbits = zeros(Bool, N)
    
    sign = 0x0
    if P[1, 2*N+1]==0
        sign = 0x0            
    elseif P[1, 2*N+1]==0.5
        sign = 0x1
    elseif P[1, 2*N+1]==1
        sign = 0x2            
    elseif P[1, 2*N+1]==1.5
        sign = 0x3            
    end
    for j in 1:N
        Xbits[j] = Bool(P[1, 2+2*(j-1)])    
        Zbits[j] = Bool(P[1, 1+2*(j-1)])
    end
    Pauli = PauliOperator(sign,Xbits,Zbits)
    return Pauli
end

function QCliffMeasure(S, P, keepRes=true)
    #println("keepRes = ", keepRes)
    if keepRes==true
        #println("keepRes true = ", keepRes)        
        newstate, anticomindex, result = project!(S, P)     
    elseif keepRes==false 
        #println("keepRes false = ", keepRes)                
        newstate, anticomindex = project!(S, P, keep_result=false)                 
    end
    deterministic = 0
    if anticomindex==0
        deterministic = 1
    else
        deterministic = 0
    end    
    if keepRes==true
        if isnothing(result)
            measureRes = rand([0x0,0x2])
            #println("measureRes = ", measureRes)
            newstate.phases[anticomindex] = measureRes
            result = measureRes
        end
        
        if result == 0x0
            measureRes = 1
        elseif result == 0x1
            measureRes = 0.5
        elseif result == 0x2
            measureRes = 2
        elseif result == 0x3
            measureRes = 1.5        
        end            
        return newstate, measureRes, deterministic
    elseif keepRes==false
        return newstate, deterministic
    end
end




function convQCliffMeasure(S, P, keepRes=true)
    #println("keepRes 2 = ", keepRes)
    tempS = copy(S)
    PPauli = convertPauliRep(P)
    #println("PPauli = ", PPauli)
    convS  = convertStabRep(tempS)
    #println("convS = ", convS)    
    if keepRes==true
        measurePauliRes = QCliffMeasure(convS, PPauli)
    else
        measurePauliRes = QCliffMeasure(convS, PPauli, keepRes)
    end
    newS = convBackStab(measurePauliRes[1])
    #println("newS = ", newS)    
    if keepRes==true       
        measureRes = measurePauliRes[2]
        #println("measureRes = ", measureRes)        
        deterministic = measurePauliRes[3]
        #println("deterministic = ", deterministic)        
        
        return newS, measureRes, deterministic
    elseif keepRes==false
        deterministic = measurePauliRes[2]        
        return newS, deterministic    
    end    

end



function QCliffTimeEvolveAncillaShort(L, p, T, withScramble, keepRes, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit 
    # which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (2*L+1)*(2L+1)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    # We remove the destabilizers from the Stabilizer matrix. 
    # We remove the sign bit from the stabilizers and random matrices. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    Ginit = zeros(Int8, 2*L+1, 4*L+3)
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    initStabDestab = initProdStateShort(2*L+1, ancilla) # Initial state is a product state along the X axis
    println("sizeof = ", sizeof(initStabDestab))
    Ginit = initStabDestab[1:2*L+1, :] # Initial state is a product state along the X axis    
    
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1
    
    
    
    GStab = zeros(size(Ginit))
    GStab = copy(Ginit)
    
    #println(convertStabRep(GStab))
    
    if size(probArray)[1] == 0
        givenRand = false
    else
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
    else
        givenU = true
    end    
    
    GStabLastGate = zeros(Int8, 2*L+1, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    
    ancillMeasure = zeros(Int8, 1, 4*L+3)
    ancillMeasure[1, 4*L+1] = 1      # Measuring vector index
    ancilMeasPauli = convertPauliRep(ancillMeasure)

    ancillaVecX = zeros(Int8, 1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)

    
    ancillaVecY = zeros(Int8, 1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(Int8, 1, 4*L+3)
    ancillaVecZ[4*L+1] = 1 
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    

    
    # Time evolution WITHOUT measurement #
    ######################################
    ######################################    
    ######################################
    ######################################   
    
    
    # We start by applying unitary time evolution without any measurements. 
    pop = [0, 1]
    weights = [1-p, p]
    
    randBit = [0, 0, 0, 0]
    
    A = [1, 4*Int(L)]
    
    
    randU = RandUnitaries()
    keepRes=false
    #println("keepRes original = ", keepRes)
    TScramble = 4*L; 
    if withScramble

        for t in 1:TScramble
            if !(givenRand)
                B = rand(Float64, 2*L)
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end
            
            if t%10==0
                #println("t = ", t)
            end 
            
            randVec = zeros(Int8, 1, 4*L+3)
    
            largeU = zeros(Int8, 4*L+2, 4*L+3)
        
            modTwo = t%2
        
            if modTwo==1
                GStabInit = copy(GStab)
                        
                for i in 1:L
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1) 
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                    
                end              
                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2            
                GStab = PauliTransform(GStabInit, largeU)
            else    
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = randIntVec1[t*L]
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                
                #randBit = rand(0:1, 4, 1)             
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1
            
                GStab = PauliTransform(GStabInit, largeU)
                #println("after PauliTransform modTwo==0")            
                #ch = checkOrtho(GStab)            
            end
        end
        #println('GStabInit \n', GStabInit)
                #println("cursor")

        #println('GStab \n', GStab)
        A = [1, 4*Int(L)]
        #println("cursor entropy")
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
        #println(GStab[1:2*L+1, :])    
        #println("entanglement before measurement = ", EE)
        #print("entanglement measure ", measureResVec)
    end    
    #println("keepRes original 2 = ", keepRes)    
    GStab1 = zeros(Int8, size(GStab))
    GStab2 = zeros(Int8, size(GStab))    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################
    ###################################    
    ###################################
    ###################################   
    #println("end conversion")
    for t in 1:T
        #println("t = ", T)
        if !(givenRand)
            B = rand(Float64, 2*L)            
            x = [Int(i<p) for i in B]
        else
            x = copy(probArray[t, :])
            #println("x = ", x)
        end   
        #println("A = ", A)
        if t%10==0
            #println('t', t)
        end
        randVec = zeros(Int8, 1, 4*L+3)
        largeU = zeros(Int8, 4*L+2, 4*L+3)
        
        #println("keepRes original 3= ", keepRes)
        for i in 1:2*L 
            #printEntropyln("i = ", i)
            if Bool(x[i])
                #println("bool = ", true)
                randVec[2*i-1] = 1                
                #println("GStab = ", GStab[1:5, 1])
                #println("randVec = ", randVec[1:5])
                measurePauliRes = convQCliffMeasure(GStab, randVec, false)
                #println("after measure = ")
                GStfab = copy(measurePauliRes[1])                    
                randVecPauli = convertPauliRep(randVec)                                        
                randVec[2*i-1] = 0
            end
        end
        # Detecting whether Ancilla is purified or not
        #println("keepRes original 4= ", keepRes)
        if purifyTime==0
            #println("keepRes original 4 inside= ", keepRes)
            #println("A in region = ", A)
            EE = Entropy(GStab[1:2*L+1, :], A)      
            #println("keepRes original 4 inside 1= ", keepRes)
            measurePauliX = convQCliffMeasure(GStab, ancillaVecX, false)
            deterX = measurePauliX[2]
            #println("keepRes original 4 inside 2= ", keepRes)
            measurePauliY = convQCliffMeasure(GStab, ancillaVecY, false)
            deterY = measurePauliY[2]
                
            measurePauliZ = convQCliffMeasure(GStab, ancillaVecZ, false)
            deterZ = measurePauliZ[2]
            deterministic = [deterX, deterY, deterZ]
            if (deterX != 0 || deterY != 0 || deterZ != 0)
                EE = Entropy(GStab[1:2*L+1, :], A)
                #println("final EE = ", EE)
                purifyTime = t
            end
        end
        #println("keepRes original 5 = ", keepRes)                
        GStabInit = copy(GStab)
        
        modTwo = t%2
        if modTwo==1
            #println("after modTwo")
"""                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
"""                
            #end
            
            
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
        else
        """            
            for i in 1:2*L
                if Bool(x[i])
                    randVec[2*i-1] = 1
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])
                    
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #convGStab  = convertStabRep(GStab)   
                    #measurePauliRes = QCliffMeasure(convGStab, randVecPauli)                                      
                    #measuredConvGStab = copy(measurePauliRes[1])
                    #GStab = convBackStab(measuredConvGStab) 
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                    randVec[2*i-1] = 0
                end
                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
                

                
            end
        """
            GStabInit = copy(GStab)                        
            
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                randbit = [0, 0, 0, 0]                
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            randbit = [0, 0, 0, 0]            
            
            #randBit = rand(0:1, 4, 1)          
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)
            """
            if purifyTime==0
                #println("inside purifyTime zero")
                intermediateGStab = copy(GStab)                                 
                convInterGStab  = convertStabRep(intermediateGStab)      
                
                
                measurePauliX = QCliffMeasure(convInterGStab, ancilVecXPauli)                                      
                deterX = measurePauliX[3]

                measurePauliY = QCliffMeasure(convInterGStab, ancilVecYPauli)                                                      
                deterY = measurePauliY[3]
                
                measurePauliZ = QCliffMeasure(convInterGStab, ancilVecZPauli)
                deterZ = measurePauliZ[3]
    
                deterministic = [deterX, deterY, deterZ]
                if (deterX != 0 || deterY != 0 || deterZ != 0)
                    purifyTime = t
                end                
            end 
            """
        end
    end
    
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)
    #println("entanglement entropy = ", EE)
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabY)          
    convFinalGStabZ  = convertStabRep(finalGStabZ)          
                
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli, false) 
    deterX = measurePauliX[2]        
    finalGStabX = measurePauliX[1]


    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli, false)
    finalGStabY = measurePauliY[1]
    deterY = measurePauliY[2]

    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli, false)    
    finalGStabZ = measurePauliZ[1]
    deterZ = measurePauliZ[2]
    
    deterministic = [deterX, deterY, deterZ]
    
        
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    if purifyTime == 0
        #purifyTime = NaN
    end
    return GStab, deterministic, purifyTime, EE 
end





function QCliffTimeEvolveAncilla(L, p, T, withScramble, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (4*L+2)*(4L+2)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    Ginit = zeros(2*L+1, 4*L+3)
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    initStabDestab = initProdState(2*L+1, ancilla) # Initial state is a product state along the X axis
    
    Ginit = initStabDestab[1:2*L+1, :] # Initial state is a product state along the X axis    
    
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1
    
    
    #Ginit[L, ancillaQbit+2] = 1   # X=1
    #Ginit[L, middleQbit] = 1 # 
    #Ginit[L, middleQbit-1] = 0 #     
    
    #Ginit[2*L+1, ancillaQbit+2] = 0 # X=0
    #Ginit[2*L+1, ancillaQbit+1] = 1 # Z=1
    #Ginit[2*L+1, middleQbit-1] = 1  # Z=1
    
    GStab = zeros(size(Ginit))
    GStab = copy(Ginit)
    
    #println(convertStabRep(GStab))
    
    if size(probArray)[1] == 0
        givenRand = false
    else
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
    else
        givenU = true
    end
    
    GStabLastGate = zeros(2*L+1, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    
    ancillMeasure = zeros(1, 4*L+3)
    ancillMeasure[1, 4*L+1] = 1      # Measuring vector index
    #println("initial conversion ")
    ancilMeasPauli = convertPauliRep(ancillMeasure)
    #println(ancilMeasPauli)
    ancillaVecX = zeros(1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)

    
    ancillaVecY = zeros(1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(1, 4*L+3)
    ancillaVecZ[4*L+1] = 1   
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    
    #println("ancilVecZPauli = ", ancilVecZPauli)
    #
    #println(ancilVecZPauli)    
    
    # Time evolution WITHOUT measurement #
    ######################################
    ######################################    
    ######################################
    ######################################   
    
    
    # We start by applying unitary time evolution without any measurements. 
    pop = [0, 1]
    weights = [1-p, p]
    
    randBit = [0, 0, 0, 0]
    
    A = [1, 4*Int(L)]
    
    
    randU = RandUnitaries()
    TScramble = Int(4*L); 
    #print("TScramble = ", TScramble)
    if withScramble

        for t in 1:TScramble
            if !(givenRand)
                B = rand(Float64, 2*L)
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end
            if t%10==0
                #println("t = ", t)
            end 
            randVec = zeros(1, 4*L+3)
    
            largeU = zeros(4*L+2, 4*L+3)
        
            modTwo = t%2
        
            if modTwo==1
                GStabInit = copy(GStab)
                        
                for i in 1:L
                    if givenU
                        randInt = Int(randIntVec1[(t-1)*L + i])
                    else
                        randInt = Int(rand(0:719))
                    end
                    #print("randInt = ", randInt)
                    #print("randU[4*randInt+1:4*randInt+4, 1:4] = ", randU[4*randInt+1:4*randInt+4, 1:4])
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1) 
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                end  
            

                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2
            
                GStab = PauliTransform(GStabInit, largeU)
            else
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = Int(randIntVec1[(t-1)*L + i])
                    else
                        randInt = Int(rand(0:719))
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = Int(randIntVec1[t*L])
                else
                    randInt = Int(rand(0:719))
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                
                #randBit = rand(0:1, 4, 1)             
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1
            
                GStab = PauliTransform(GStabInit, largeU)
                #println("after PauliTransform modTwo==0")            
                #ch = checkOrtho(GStab)
            
            end
        end
        #println('GStabInit \n', GStabInit)
                #println("cursor")

        #println('GStab \n', GStab)
        A = [1, 4*Int(L)]
        #println("cursor entropy")
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
        #println(GStab[1:2*L+1, :])    
        #println("entanglement before measurement = ", EE)
        #print("entanglement measure ", measureResVec)
    end    
        
    GStab1 = zeros(size(GStab))
    GStab2 = zeros(size(GStab))    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################
    ###################################    
    ###################################
    ###################################   
    #println("end conversion")
    for t in 1:T
        
        if !(givenRand)
            B = rand(Float64, 2*L)            
            x = [Int(i<p) for i in B]
        else
            x = copy(probArray[t, :])
            #println("x = ", x)
        end

        if t%10==0
            #println('t', t)
        end
        randVec = zeros(1, 4*L+3)
        largeU = zeros(4*L+2, 4*L+3)
        
        
        for i in 1:2*L
            if Bool(x[i])
                randVec[2*i-1] = 1
                
                measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                GStab = copy(measurePauliRes[1])  
                #println("GStab = ", size(GStab))
                measureResVec[t, i] = copy(measurePauliRes[2])
                
                randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #convGStab  = convertStabRep(GStab)   
                    #measurePauliRes = QCliffMeasure(convGStab, randVecPauli)                                      
                    #measuredConvGStab = copy(measurePauliRes[1])
                    #GStab = convBackStab(measuredConvGStab) 
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                randVec[2*i-1] = 0
            end
        end
        # Detecting whether Ancilla is purified or not
        
        if purifyTime==0
            EE = Entropy(GStab[1:2*L+1, :], A)
            #println("purifyTime==0 EE = ", EE)                
            measurePauliX = convQCliffMeasure(GStab, ancillaVecX)
            deterX = measurePauliX[3]

            measurePauliY = convQCliffMeasure(GStab, ancillaVecY)
            deterY = measurePauliY[3]
                
            measurePauliZ = convQCliffMeasure(GStab, ancillaVecZ)
            deterZ = measurePauliZ[3]
            deterministic = [deterX, deterY, deterZ]
            if (deterX != 0 || deterY != 0 || deterZ != 0)
                EE = Entropy(GStab[1:2*L+1, :], A)
                #println("final EE = ", EE)
                purifyTime = t
            end
        end
                        
        GStabInit = copy(GStab)
        
        modTwo = t%2
        if modTwo==1
            #println("after modTwo")
"""                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
"""                
            #end
            
            
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = Int(rand(0:719))
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
        else
        """            
            for i in 1:2*L
                if Bool(x[i])
                    randVec[2*i-1] = 1
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])
                    
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #convGStab  = convertStabRep(GStab)   
                    #measurePauliRes = QCliffMeasure(convGStab, randVecPauli)                                      
                    #measuredConvGStab = copy(measurePauliRes[1])
                    #GStab = convBackStab(measuredConvGStab) 
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                    randVec[2*i-1] = 0
                end
                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
                

                
            end
        """
            GStabInit = copy(GStab)                        
            
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                randbit = [0, 0, 0, 0]                
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            randbit = [0, 0, 0, 0]            
            
            #randBit = rand(0:1, 4, 1)          
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)
            """
            if purifyTime==0
                #println("inside purifyTime zero")
                intermediateGStab = copy(GStab)                                 
                convInterGStab  = convertStabRep(intermediateGStab)      
                
                
                measurePauliX = QCliffMeasure(convInterGStab, ancilVecXPauli)                                      
                deterX = measurePauliX[3]

                measurePauliY = QCliffMeasure(convInterGStab, ancilVecYPauli)                                                      
                deterY = measurePauliY[3]
                
                measurePauliZ = QCliffMeasure(convInterGStab, ancilVecZPauli)
                deterZ = measurePauliZ[3]
    
                deterministic = [deterX, deterY, deterZ]
                if (deterX != 0 || deterY != 0 || deterZ != 0)
                    purifyTime = t
                end                
            end 
            """
        end
    end
    
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)
    #println("entanglement entropy = ", EE)
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabY)          
    convFinalGStabZ  = convertStabRep(finalGStabZ)          
    
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]        
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]

    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]

    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    
    deterministic = [deterX, deterY, deterZ]
    #println("deterministic = ", deterministic)
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    if purifyTime == 0
        #purifyTime = NaN
    end
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, EE 
    
end





function randArrGeneration(L, p, T, Ncircuit, Nbatch, Scramble)
        

    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end
    println("ScrambleLabel = ", ScrambleLabel)
    println("Nb = ", Nbatch)
    if Nbatch==false
        Nbatch = 1
    end
    for n in 1:Nbatch
        print("n = ", n)
        randVecArr1 = zeros(Ncircuit, L*T);
        randVecArr2 = zeros(Ncircuit, L*T);
        probArr = zeros(T, 2*L);
        probArrMat = zeros(Ncircuit, T, 2*L);
    
        for c in 1:Ncircuit
            if c%100 == 0
                println("c = ", c)
            end
            randVec1 = rand(0:719, 1, L*T)
            randVec2 = rand(0:719, 1, L*T)
            randVecArr1[c, :] = copy(randVec1[1, 1:L*T])
            randVecArr2[c, :] = copy(randVec2[1, 1:L*T]) 
            for t in 1:T
                B = rand(Float64, 2*L)    
                for i in 1:2*L
                    probArr[t, i] = Int(B[i]<p)
                end
            end
            probArrMat[c, :, :] = copy(probArr)
        end
        
        #save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")


        rshpProb = reshape(probArrMat, (Ncircuit*T, 2*L))
        #dblRshpVec = reshape(rshpVec, (Ncircuit, 2*L))
        
        #CSV.write("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(rshpProb), writeheader=true)
        if Nbatch>1  
            print("Nb>1")            
            try
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
            catch y                
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
                end
            end
        elseif (Nbatch ==1) || (Nbatch == false)
            print("elseif")
            try            
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                #CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                #CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
            catch y
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                    #CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                    #CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
                end
            end
        end

        
"""
        if Nbatch>1
            save("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$n$ScrambleLabel.jld", "data", probArrMat)
            save("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$n$ScrambleLabel.jld", "data", randVecArr1)
            save("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$n$ScrambleLabel.jld", "data", randVecArr2)
            println("probArrMatL$L-Ncirc$Ncircuit$ScrambleLabel.jld")            
        elseif Nbatch ==1 
            save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
            save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
            save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)
            CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
            CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
            println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$n$ScrambleLabel.jld")
        end  
"""                
    end
    
end    






function QCliffTimeEvolveAncilla(L, p, T, withScramble, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (4*L+2)*(4L+2)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    Ginit = zeros(2*L+1, 4*L+3)
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    initStabDestab = initProdState(2*L+1, ancilla) # Initial state is a product state along the X axis
    
    Ginit = initStabDestab[1:2*L+1, :] # Initial state is a product state along the X axis    
    
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1
    
    
    #Ginit[L, ancillaQbit+2] = 1   # X=1
    #Ginit[L, middleQbit] = 1 # 
    #Ginit[L, middleQbit-1] = 0 #     
    
    #Ginit[2*L+1, ancillaQbit+2] = 0 # X=0
    #Ginit[2*L+1, ancillaQbit+1] = 1 # Z=1
    #Ginit[2*L+1, middleQbit-1] = 1  # Z=1
    
    GStab = zeros(size(Ginit))
    GStab = copy(Ginit)
    
    #println(convertStabRep(GStab))
    
    if size(probArray)[1] == 0
        givenRand = false
    else
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
    else
        givenU = true
    end    
    
    GStabLastGate = zeros(2*L+1, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    
    ancillMeasure = zeros(1, 4*L+3)
    ancillMeasure[1, 4*L+1] = 1      # Measuring vector index
    #println("initial conversion ")
    ancilMeasPauli = convertPauliRep(ancillMeasure)
    #println(ancilMeasPauli)
    ancillaVecX = zeros(1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)

    
    ancillaVecY = zeros(1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(1, 4*L+3)
    ancillaVecZ[4*L+1] = 1  
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    
    #println("ancilVecZPauli = ", ancilVecZPauli)
    #
    #println(ancilVecZPauli)    
    
    # Time evolution WITHOUT measurement #
    ######################################
    ######################################    
    ######################################
    ######################################   
    
    
    # We start by applying unitary time evolution without any measurements. 
    pop = [0, 1]
    weights = [1-p, p]
    
    randBit = [0, 0, 0, 0]
    
    A = [1, 4*Int(L)]
    
    
    randU = RandUnitaries()
    TScramble = 4*L; 
    if withScramble

        for t in 1:TScramble
            if !(givenRand)
                B = rand(Float64, 2*L)
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end
            if t%10==0
                #println("t = ", t)
            end 
            randVec = zeros(1, 4*L+3)
    
            largeU = zeros(4*L+2, 4*L+3)
        
            modTwo = t%2
        
            if modTwo==1
                GStabInit = copy(GStab)
                        
                for i in 1:L
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1) 
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                end  
            

                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2
            
                GStab = PauliTransform(GStabInit, largeU)
            else
    
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = randIntVec1[t*L]
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                
                #randBit = rand(0:1, 4, 1)             
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1
            
                GStab = PauliTransform(GStabInit, largeU)
                #println("after PauliTransform modTwo==0")            
                #ch = checkOrtho(GStab)
            
            end
        end
        #println('GStabInit \n', GStabInit)
                #println("cursor")

        #println('GStab \n', GStab)
        A = [1, 4*Int(L)]
        #println("cursor entropy")
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
        #println(GStab[1:2*L+1, :])    
        #println("entanglement before measurement = ", EE)
        #print("entanglement measure ", measureResVec)
    end    
        
    GStab1 = zeros(size(GStab))
    GStab2 = zeros(size(GStab))    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################
    ###################################    
    ###################################
    ###################################   
    #println("end conversion")
    for t in 1:T
        if !(givenRand)
            B = rand(Float64, 2*L)            
            x = [Int(i<p) for i in B]
        else
            x = copy(probArray[t, :])
            #println("x = ", x)
        end        
        #println("x = ", x)
        if t%10==0
            #println('t', t)
        end
        randVec = zeros(1, 4*L+3)
        largeU = zeros(4*L+2, 4*L+3)
        
        
        for i in 1:2*L 
            if Bool(x[i])
                randVec[2*i-1] = 1
                
                measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                GStab = copy(measurePauliRes[1])                    
                measureResVec[t, i] = copy(measurePauliRes[2])
                    
                randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #convGStab  = convertStabRep(GStab)   
                    #measurePauliRes = QCliffMeasure(convGStab, randVecPauli)                                      
                    #measuredConvGStab = copy(measurePauliRes[1])
                    #GStab = convBackStab(measuredConvGStab) 
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                randVec[2*i-1] = 0
            end
        end
        # Detecting whether Ancilla is purified or not
        
        if purifyTime==0
            #println("keepRes original 4 inside= ", false)
            #println("A in region = ", A)

            EE = Entropy(GStab[1:2*L+1, :], A)
            #println("purifyTime==0 EE = ", EE)  
            tempGStabX = copy(GStab)
            tempGStabY = copy(GStab)
            tempGStabZ = copy(GStab)
            
            measurePauliX = convQCliffMeasure(tempGStabX, ancillaVecX)
            deterX = measurePauliX[3]

            measurePauliY = convQCliffMeasure(tempGStabY, ancillaVecY)
            deterY = measurePauliY[3]
                
            measurePauliZ = convQCliffMeasure(tempGStabZ, ancillaVecZ)
            deterZ = measurePauliZ[3]
            deterministic = [deterX, deterY, deterZ]
            if (deterX != 0 || deterY != 0 || deterZ != 0)
                EE = Entropy(GStab[1:2*L+1, :], A)
                #println("final EE = ", EE)
                purifyTime = t
            end
        end
                        
        GStabInit = copy(GStab)
        
        modTwo = t%2
        if modTwo==1
            #println("after modTwo")
"""                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
"""                
            #end
            
            
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
        else
        """            
            for i in 1:2*L
                if Bool(x[i])
                    randVec[2*i-1] = 1
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])
                    
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #convGStab  = convertStabRep(GStab)   
                    #measurePauliRes = QCliffMeasure(convGStab, randVecPauli)                                      
                    #measuredConvGStab = copy(measurePauliRes[1])
                    #GStab = convBackStab(measuredConvGStab) 
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                    randVec[2*i-1] = 0
                end
                
                if Bool(x[i])
                    randVec[2*i-1] = 1         # Z measurement, not an X measurement
                    randVecPauli = convertPauliRep(randVec)                                        
                    #println("randVec = ", randVecPauli)                    
                    #println("GStab before measurement = ", GStab)
                    GStab1 = copy(GStab)
                    measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #println("GStab after measurement = ", GStab)    
                    GStab2 = copy(GStab)
                    #println(GStab2 == GStab1)
                    GStab = copy(measurePauliRes[1])                    
                    measureResVec[t, i] = copy(measurePauliRes[2])                                                                                
                    randVec[2*i-1] = 0                                                    
                end
                

                
            end
        """
            GStabInit = copy(GStab)                        
            
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                randbit = [0, 0, 0, 0]                
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            randbit = [0, 0, 0, 0]            
            
            #randBit = rand(0:1, 4, 1)          
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)
            """
            if purifyTime==0
                #println("inside purifyTime zero")
                intermediateGStab = copy(GStab)                                 
                convInterGStab  = convertStabRep(intermediateGStab)      
                
                
                measurePauliX = QCliffMeasure(convInterGStab, ancilVecXPauli)                                      
                deterX = measurePauliX[3]

                measurePauliY = QCliffMeasure(convInterGStab, ancilVecYPauli)                                                      
                deterY = measurePauliY[3]
                
                measurePauliZ = QCliffMeasure(convInterGStab, ancilVecZPauli)
                deterZ = measurePauliZ[3]
    
                deterministic = [deterX, deterY, deterZ]
                if (deterX != 0 || deterY != 0 || deterZ != 0)
                    purifyTime = t
                end                
            end 
            """
        end
    end
    
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)
    #println("entanglement entropy = ", EE)
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabX)          
    convFinalGStabZ  = convertStabRep(finalGStabX)          
                
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]    
    
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]

    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]

    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    
    deterministic = [deterX, deterY, deterZ]
    #println("deterministic = ", deterministic)
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    if purifyTime == 0
        #purifyTime = NaN
    end
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, EE 
end

function randArrGenerationARGS(args)
    
#julia JuliaCliffordMainModulesRandArrGeneration.jl L p T Ncircuit nb cVec(false) Ncirc1 Ncirc2
    
    
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    #Nbatch = false
    Nbatch = parse(Int, args[5])    
    
    println("Nbatch = ", Nbatch)
    
    nb = parse(Int, args[6])
    
    #severalJob = false    
    #severalJob = parse(Bool, args[7])
    #println("severalJOB = ", severalJob)
    #if severalJob ==true
    #    job_id=ENV["SLURM_ARRAY_TASK_ID"]
    #end
    cVec = false
    try
        cVec = parse(Int, args[7])
        catch y.
        if isa(y, ArgumentError)
            cVec = false
        end
    end
    
    Ncirc1 = parse(Int, args[8])
    Ncirc2 = parse(Int, args[9])    
    
    Scramble = parse(Int, args[10])
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    
    for n in 1:Nbatch
        randVecArr1 = zeros(Ncircuit, L*T);
        randVecArr2 = zeros(Ncircuit, L*T);
        probArr = zeros(T, 2*L);
        probArrMat = zeros(Ncircuit, T, 2*L);
    
        for c in 1:Ncircuit
            if c%100 == 0
                println("c = ", c)
            end
            randVec1 = rand(0:719, 1, L*T)
            randVec2 = rand(0:719, 1, L*T)
            randVecArr1[c, :] = copy(randVec1[1, 1:L*T])
            randVecArr2[c, :] = copy(randVec2[1, 1:L*T])    
            for t in 1:T
                B = rand(Float64, 2*L)    
                for i in 1:2*L
                    probArr[t, i] = Int(B[i]<p)
                end
            end
            probArrMat[c, :, :] = copy(probArr)
        end

        #save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")


        rshpProb = reshape(probArrMat, (Ncircuit*T, 2*L))
        #dblRshpVec = reshape(rshpVec, (Ncircuit, 2*L))
    
        #CSV.write("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(rshpProb), writeheader=true)
        if Nbatch>1  
            try
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
            catch y                
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
                end
            end
        elseif (Nbatch ==1) || (Nbatch == false)
            try            
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
            catch y
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                    CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                    CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
                end
            end
        end
        #println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$n$ScrambleLabel.jld")
    end
end    








function TimeEvolveAncilla(L, p, T, withScramble, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (4*L+2)*(4L+2)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    randU = RandUnitaries()
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    Ginit = initProdState(2*L+1, ancilla) # Initial state is a product state along the Z axis
    #println('(Ginit) \n', (Ginit))
    
    # Odd: Z operators, Even: X operators.
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1

    #Ginit[3*L+1, middleQbit+2] = 0  # X=0
    #Ginit[3*L+1, middleQbit+1] = 1  # Z=1
    
    Ginit[4*L+2, ancillaQbit+2] = 0 # X=0
    Ginit[4*L+2, ancillaQbit+1] = 1 # Z=1

    for i in 1:size(Ginit)[1]
        #println("Ginit[$i] = ", Ginit[i, :])
    end    
    #ch = checkOrtho(Ginit)    

    #println("Ginit = \n")
    #printMatrix(Ginit)
    #print('Sinit \n', size(Sinit))
    GStab = zeros(size(Ginit))
    #print('size(Ginit) \n', Ginit)
    GStab = copy(Ginit)
    #println(convertStabRep(GStab))
    #println(convertStabRep(GStab[2*L+2:4*L+2, :]))
    
    if size(probArray)[1] == 0
        givenRand = false
        #println("No Given Measure Array")    
    else
        #println("size(probArray)", size(probArray))
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
        #println("No U")    
    else
        givenU = true
        #println("U Given")    
    end    
    #println("givenRand = ", givenRand)
    
    GStabLastGate = zeros(4*L+2, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    
    ancillMeasure = zeros(1, 4*L+3)
    ancillMeasure[4*L+1] = 1      
    #println("ancillMeasure = ", ancillMeasure)

    ancillaVecX = zeros(1, 4*L+3)
    ancillaVecX[4*L+2] = 1 

    ancillaVecY = zeros(1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    
    ancillaVecZ = zeros(1, 4*L+3)
    ancillaVecZ[4*L+1] = 1     
    
    # Time evolution WITHOUT measurement #
    ######################################
    ######################################    
    ######################################
    ######################################   
    
    
    # We start by applying unitary time evolution without any measurements. 
    pop = [0, 1]
    weights = [1-p, p]
    

    randBit = [0, 0, 0, 0]
    TScramble = 4*L; 
    if withScramble

        for t in 1:TScramble
            if !(givenRand)
                A = rand(Float64, 2*L)
            #println("size(A) = ", size(A))
            #A[2L] = 0
                x = [Int(i<p) for i in A]
            else
            #println("probArray[t, :]", size(probArray[t, :]))
                x = copy(probArray[t, :])
            end
            if t%10==0
                #println("t = ", t)
            end 
            randVec = zeros(1, 4*L+3)
    
            largeU = zeros(4*L+2, 4*L+3)
        
            modTwo = t%2
        
            if modTwo==1
                GStabInit = copy(GStab)
                        
                for i in 1:L
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                #println("randU[1, :] = ", randU[1, :])
                #println("randU[2, :] = ", randU[2, :])                
                #println("randU[3, :] = ", randU[3, :])
                #println("randU[4, :] = ", randU[4, :])                
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1) 
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                end  
            

                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
     #           println("size largeU", size(largeU))
    #            println("size GStabInit", size(GStabInit))            
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2
                
                GStab = PauliTransform(GStabInit, largeU)
                #println("after PauliTransform modTwo==1")
                #ch = checkOrtho(GStab)
                #println('ch in unitary modTwo \n', ch)
            else
                            #println("cursor else")
                
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    #randBit = rand(0:1, 4, 1)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = randIntVec1[t*L]
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                
                #randBit = rand(0:1, 4, 1)             
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1
                
                GStab = PauliTransform(GStabInit, largeU)
                #println("after PauliTransform modTwo==0")            
                #ch = checkOrtho(GStab)
                
            end
        end
        #println('GStabInit \n', GStabInit)
                #println("cursor")

        #println('GStab \n', GStab)
        A = [1, 4*Int(L)]
        #println("cursor entropy")
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
        #println(GStab[1:2*L+1, :])    
        #println("entanglement before measurement = ", EE)
        #print("entanglement measure ", measureResVec)
    end    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################
    ###################################    
    ###################################
    ###################################   
    
    for t in 1:T

        if !(givenRand)
            A = rand(Float64, 2*L)
            #println("size(A) = ", size(A))
            
            x = [Int(i<p) for i in A]
            #println("after x assignment")
        else
            #println("probArray[t, :]", size(probArray[t, :]))
            x = copy(probArray[t, :])
        end        
        if t%10==0
            #println('t', t)
        end
        randVec = zeros(1, 4*L+3)
        largeU = zeros(4*L+2, 4*L+3)
        modTwo = t%2
        
        if modTwo==1
            #println("after modTwo")
            for i in 1:2*L
                #println("i, x[i] = ", i, x[i])

                if Bool(x[i])
                    randVec[2*i-1] = 1
                    #println("after randVec")
                    measurePauliRes = measurePauli(randVec, GStab)
                    #println("after measurePauli")
                    GStab = copy(measurePauliRes[1])
                    #println("after measurement modTwo==1")            
                    #ch = checkOrtho(GStab)
                    
                    #println("after copy")                    
                    measureResVec[t, i] = copy(measurePauliRes[2]);
                    #println("after copy 2")                         
                    #ch = checkOrtho(GStab)
                    #println("ch in measure modTwo = ", ch)
                    #println(randVec)
                    randVec[2*i-1] = 0;
                end
            end
            #println(convertStabRep(GStab))
            #println('\n')
            # Detecting whether Ancilla is purified or not
            if purifyTime==0
                #println("inside purifyTime zero")
                intermediateGStab = copy(GStab)
                measurePauliX = measurePauli(ancillaVecX, intermediateGStab)        
                deterX = measurePauliX[3]

                measurePauliY = measurePauli(ancillaVecY, intermediateGStab)        
                deterY = measurePauliY[3]

                measurePauliZ = measurePauli(ancillaVecZ, intermediateGStab)        
                deterZ = measurePauliZ[3]
    
                deterministic = [deterX, deterY, deterZ]
                if (deterX != 0 || deterY != 0 || deterZ != 0)
                    purifyTime = t
                end
            #else
            #    println("inside purifyTime not zero")
            end
            
            
            
            
            #measureAncilla = measurePauli(ancillMeasure, GStab)
            #determAncilla = measureAncilla[3]
            #println("determAncilla = ", determAncilla)
            #if Bool(determAncilla)
                
            #end
            
            GStabInit = copy(GStab)
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            #println("after largeU")
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            #println('largeU \n', largeU)
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
            #println("after PauliTransform modTwo==1")            
            #ch = checkOrtho(GStab)
            
            #ch = checkOrtho(GStab)
            #println('ch in unitary modTwo \n', ch)
        else
            
            for i in 1:2*L
                #print('i, x[i]', i, x[i])
                
                if Bool(x[i])
                    randVec[2*i-1] = 1
                    #ch = checkOrtho(GStab)
                    #println('ch in Before measure modTwo 0\n', ch)
                    
                    measurePauliRes = measurePauli(randVec, GStab)
                    GStab = copy(measurePauliRes[1])
                    #println("after measurement modTwo==0")            
                    #ch = checkOrtho(GStab)
                    
                    measureResVec[t, i] = copy(measurePauliRes[2])
                    
                    #println('measureRes', measureRes)
                    #ch = checkOrtho(GStab)
                    #println('ch in measure modTwo 0 \n', ch)
                    
                    randVec[2*i-1] = 0
                end
            end
            #println("GStabInit = copy(GStab) ")
            GStabInit = copy(GStab)
            
            
            if purifyTime==0
                #println("inside purifyTime zero")
                intermediateGStab = copy(GStab)
                measurePauliX = measurePauli(ancillaVecX, intermediateGStab)        
                deterX = measurePauliX[3]

                measurePauliY = measurePauli(ancillaVecY, intermediateGStab)        
                deterY = measurePauliY[3]

                measurePauliZ = measurePauli(ancillaVecZ, intermediateGStab)        
                deterZ = measurePauliZ[3]
    
                deterministic = [deterX, deterY, deterZ]
                if (deterX != 0 || deterY != 0 || deterZ != 0)
                    purifyTime = t
                end
                
            #else
            #    println("inside purifyTime not zero")
                 
            end
            
            
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                #randInt = random.randint(720)
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            
            
            #randBit = rand(0:1, 4, 1)          
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)
            #println("after PauliTransform modTwo==1")            
            #ch = checkOrtho(GStab)
            #println('ch in unitary modTwo 0\n', ch)
        end
    end
    
    #print('Final GStab \n', GStab)
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)
    #println(convertStabRep(GStab))
    #println("GStab = ", GStab[1:2*L+1, :])    
    #println("entanglement after measurement = ", EE)
    

    
    #println('ancillaVecZ\n', ancillaVecZ)
    finalGStab1 = copy(GStab)
    finalGStab2 = copy(GStab)
    finalGStab3 = copy(GStab)    
    measurePauliX = measurePauli(ancillaVecX, finalGStab1)        
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]
    
    measurePauliY = measurePauli(ancillaVecY, finalGStab2)        
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]
    
    measurePauliZ = measurePauli(ancillaVecZ, finalGStab3)        
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    
    deterministic = [deterX, deterY, deterZ]
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    #println("deterministic $deterX $deterY $deterZ")
    #println("measurements $finalMeasureX $finalMeasureY $finalMeasureZ")
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    #println("purifyTime = ", purifyTime)
    #println('finalMeasureRes', finalMeasureRes)
    if purifyTime == 0
        #purifyTime = NaN
    end
    
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, EE
end

#A = randArrGenerationARGS(ARGS)

function determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch=false, nb=false, cVec = false, csv=false)
#    determinePurifyTime
    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end
    
    if csv
        rv1 = zeros(Ncircuit, L*T);""
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv1 = zeros(size(df, 1), size(df, 2));

        println("size rv1 = ", size(rv1), size(rv2))

        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        println("rv1[1, 1:10] = ", rv1[1, 1:10])
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1] 
            #println("i = ", i)        
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end

    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    Ncirc1 = 1
    Ncirc2 = Ncircuit
    purifyTimeVec = zeros(Ncircuit)

    #Scramble = parse(Int, args[10])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    keepTraj = true
    if Nbatch==false || Nbatch == 1
        try                
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        catch y                
            if isa(y, ArgumentError)
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                
            end
        end
            
    else
        println("nb = ", nb)
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]       
    end
    
    println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")    
    save("probArrMatDeterminePTL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", probArrMat)
    println("loop")

    if cVec==false
        Threads.@threads for c in Ncirc1:Ncirc2
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            #A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
            #    rv2[c, :], probArrMat[c, :, :])            
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])

            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end            
    else 
        Threads.@threads for c in cVec
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            #A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
            #    rv2[c, :], probArrMat[c, :, :]);   
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            

            flush(stdout)
            println("c = ", c, " pure time = ", A[5])
            println("final measure = ", A[3])
            println("determ = ", A[4])            
            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end
    end
    print(transpose(purifyTimeVec))
    #    determinePurifyTime
    #A = DataFrame(transpose(purifyTimeVec))
    if cVec==false 
        if Nbatch==false || Nbatch == 1
            if Ncirc1==1
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld")
                #println("Ncirc1!=1 || Ncirc2!=Ncircuit")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                 
            end
        elseif Nbatch!=false 
            if Ncirc1==1 && Ncirc2==Ncircuit
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)            
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                 
            end 
        end
       # println("purifyTimeVec", purifyTimeVec)
    else 
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.jld", "data", purifyTimeVec)                
        CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                         
    end

    
    
    return purifyTimeVec 
    
    
    
    """

    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T); 
    rv2 = zeros(Ncircuit, L*T); 
    Ncirc1 = 1
    Ncirc2 = Ncircuit
    purifyTimeVec = zeros(Ncircuit)
    if nbatch==false        
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]        
    else 
        println("nbatch = ", nbatch)       
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$nbatch$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$nbatch$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$nbatch$ScrambleLabel.jld")["data"]        
    
    end



    
    println("loop")


    Threads.@threads for c in Ncirc1:Ncirc2
        if c%1==0
            println("c = ", c)
        end
        A = QCliffTimeEvolveAncilla(L, p, T, false, rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
        println("c = ", c, " pure time = ", A[5])
        purifyTimeVec[c] = A[5]
        #print(purifyTimeVec[c])
    end
    if nbatch==false
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)    
    else 
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$nbatch$ScrambleLabel.jld", "data", purifyTimeVec)  
    end
    return purifyTimeVec 
    """
end




function determinePurifyTimeARGS(args)
    
#    (nb=false, cVec = false, csv=false)
    
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[5]) = ", parse(Int, args[5]))
        Nbatch = parse(Int, args[5])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end    
    println("Nbatch = ", Nbatch)
    
    nb = parse(Int, args[6])
    
    #severalJob = false
    #severalJob = parse(Bool, args[7])
    #println("severalJOB = ", severalJob)
    #if severalJob ==true
    #    job_id=ENV["SLURM_ARRAY_TASK_ID"]
    #end
    cVec = false
    try
        cVec = parse(Int, args[7])
    catch y
        if isa(y, ArgumentError)
            cVec = false
        end
    end
    
    Ncirc1 = parse(Int, args[8])
    Ncirc2 = parse(Int, args[9])    
    
    Scramble = parse(Int, args[10])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    #Ncirc1 = 1
    #Ncirc2 = Ncircuit
    purifyTimeVec = zeros(Ncircuit)

    if Nbatch==false || Nbatch == 1
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    else
        println("nb = ", nb)
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
        save("probArrMatDeterminePTL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", probArrMat)        
    end
    
    println(size(rv1))
    println(size(rv2))
    println(size(probArrMat))
    
    println("loop")
    println("cVec = ", cVec)
    
    if cVec==false
        Threads.@threads for c in Ncirc1:Ncirc2
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            #A = QCliffordAncilla(L, p, TEvolve, Bool(Scramble), keepTraj, rv1[c, :], 
            #    rv2[c, :], probArrMat[c, :, :])            
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])

            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end            
    else 
        Threads.@threads for c in cVec
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])
            println("final measure = ", A[3])
            println("determ = ", A[4])            
            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end
    end
    if cVec==false 
        if Nbatch==false || Nbatch == 1
            if Ncirc1==1
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld")
                #println("Ncirc1!=1 || Ncirc2!=Ncircuit")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)
            end
        elseif Nbatch!=false 
            if Ncirc1==1 && Ncirc2==Ncircuit
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)            
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                 
            end 
        end
       # println("purifyTimeVec", purifyTimeVec)
    else 
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.jld", "data", purifyTimeVec)                
        CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                         
    end

    return purifyTimeVec 
end 


#A = determinePurifyTimeARGS(ARGS)


function circIndArrGeneration(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT)    
    
    #Scramble = parse(Int, args[10])    
    println("Scramble = ", Scramble)
    ScrambleLabel = ""
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
     
    
    if Nbatch==false
        purifyTimeVec = zeros(Ncircuit)
        #purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    elseif Nbatch>1
        purifyTimeVec = zeros(Ncircuit*Nbatch)
        for n = 1:Nbatch
            purifyTime = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld")["data"]
            println("purifyTime = ", purifyTime)
            if n==1
                purifyTimeVec = vcat(purifyTime)
            elseif n>1
                purifyTimeVec = vcat(purifyTimeVec, purifyTime)
            end
        end
    end
    
    println("purifyTimeVec = ", purifyTimeVec)
    #NpureT = 10; # we take the first 10 purification times for learning.
    #NcircPT = 20; # we consider 10 circuit realization for each purification time.
    circIndArr = zeros(Int64, NcircPT, NpureT);
    lastFoundInd = zeros(NpureT)
    for i in 1:size(purifyTimeVec)[1]

        
#0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 
#3.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 1.0, 1.0, 
#3.0, 17.0, 5.0, 1.0, 2.0, 2.0, 3.0        
        
        if purifyTimeVec[i] <= NpureT && purifyTimeVec[i] > 0
            #println(purifyTimeVec[i], ' ', i)
            lastFoundInd[Int(purifyTimeVec[i])] += 1
            if lastFoundInd[Int(purifyTimeVec[i])] <= NcircPT
                #println("inside if")
                circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])] = Int(i)
                println("i, purifyTimeVec[i],  circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), 
                    Int(purifyTimeVec[i])] = ", i, ' ', purifyTimeVec[i], ' ', 
                    circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])])
            end
        else
            continue
        end
    end

    println("circIndArr[:, PT] = ", circIndArr[:, :])
    if Nbatch==false
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)
    elseif Nbatch>1
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)        
    end
    return circIndArr
end
#f = open("twoUnitaries.txt", "r")


        


function circIndArrGenerationARGS(args)
    
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[5]) = ", parse(Int, args[5]))
        Nbatch = parse(Int, args[5])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end    
    println("Nbatch = ", Nbatch)
    
    
    NpureT = parse(Int, args[6]);
    NcircPT = parse(Int, args[7]);
    
    Scramble = parse(Int, args[8])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    
"""
    df = CSV.read("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv", DataFrame)
    purifyTimeVec = zeros(Ncircuit)    
    println(size(df, 2))
    for i in 1:size(df, 2) 
        purifyTimeVec[i] = df[1, i]
    end

    #if Nbatch==false
    #    circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                        
    #else 
    #    circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")["data"]
    #    println("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")            
    #end    
    save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
"""        
    
    if Nbatch==false
        purifyTimeVec = zeros(Ncircuit)
        #purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    elseif Nbatch>1
        purifyTimeVec = zeros(Ncircuit*Nbatch)
        for n = 1:Nbatch
            purifyTime = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld")["data"]
            println("purifyTime = ", purifyTime)
            if n==1
                purifyTimeVec = vcat(purifyTime)
            elseif n>1
                purifyTimeVec = vcat(purifyTimeVec, purifyTime)
            end
        end
    end
    
    println("purifyTimeVec = ", purifyTimeVec)
    #NpureT = 10; # we take the first 10 purification times for learning.
    #NcircPT = 20; # we consider 10 circuit realization for each purification time.
    circIndArr = zeros(Int64, NcircPT, NpureT);
    lastFoundInd = zeros(NpureT)
    for i in 1:size(purifyTimeVec)[1]
        #println(purifyTimeVec[i])
        if purifyTimeVec[i] <= NpureT && purifyTimeVec[i] > 0
            #println(purifyTimeVec[i], ' ', i)
            lastFoundInd[Int(purifyTimeVec[i])] += 1
            if lastFoundInd[Int(purifyTimeVec[i])] <= NcircPT
                #println("inside if")
                circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])] = Int(i)
            end
        else
            continue
        end
    end

    println("circIndArr[:, PT] = ", circIndArr[:, :])
    if Nbatch==false
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)                
    elseif Nbatch>1
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)                
    end
    return circIndArr
    
end

#T=2
#Nsamp

"""
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 56, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 82, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 177, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 179, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 229, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 295, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 614, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 653, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 759, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 821, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 958, false, false)
C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 976, false, false)
#C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 295, false, false)
#C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 295, false, false)
#C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 295, false, false)
#C = createMeasureRec(L, p, T, Scramble, 10000, Ncircuit, Nbatch, NpureT, NcircPT, 295, false, false)
"""

#A = circIndArrGenerationARGS(ARGS)



function randu2(fixedSign=false, t=[], LHalf=[], i=[], randIntVec=[])
    givenU = false
    #println("givenU1 = ", givenU)
    if size(randIntVec)[1] == 0
        givenU = false
    else
        givenU = true
    end    
    
    #println("givenU2 = ", givenU)
    
    if givenU

        randInt = 1+Int(randIntVec[(t-1)*LHalf + i])
        #println("randInt = ", randInt)        
    else
        randInt = 1+rand(0:719)
    end
    
    #println("givenU3 = ", givenU)
    #randU = RandUnitaries()   
    #println("givenU4 = ", givenU)
    #u = randU[4*randInt+1:4*randInt+4, 1:4]
    
    
    
    #println("givenU5 = ", givenU)    
    #println("u = ", u)
    if givenU!=false
        u = ru2[randInt]
    elseif givenU==false
        u=rand(ru2)
    end
    #println(u)
    #println("givenU6 = ", givenU)
    #println("u = ", u[:, 1])
    #println(returnPaul(u[:,1],0x0))
    #println("u = ", u[:, 2])
    #println(returnPaul(u[:,2],0x0))    
    #println("u = ", u[:, 3])
    #println(returnPaul(u[:,3],0x0))        
    #println("u = ", u[:, 4])    
    #println(returnPaul(u[:,4],0x0))
    
    if fixedSign
        return CliffordOperator(Stabilizer([returnPaul(u[:,i],0x0) for i in 1:size(u,2)]))
    else 
        
        #println("Cliff = ", CliffordOperator(Stabilizer([returnPaul(u[:,i],rand([0x0,0x2])) for i in 1:size(u,2)])))
        return CliffordOperator(Stabilizer([returnPaul(u[:,i],rand([0x0,0x2])) for i in 1:size(u,2)]))
    end
end




function QCliffordAncilla(L, p, T, withScramble, keepTraj=false, randIntVec1 = [], randIntVec2 = [], probArray = [])
    
    # We only use the functions from the QuantumClifford Package. 
    # MiddleInd is different than the choice considered in QCliffTimeEvolveAncilla. 
    # Here MidInd=2*L, while in QCliffTimeEvolveAncilla is MidInd=L. 
    #println("Int(L/2)")
    givenU1 = false
    #println("givenU1 = ", givenU)
    if size(randIntVec1)[1] == 0
        givenU1 = false
    else
        givenU1 = true
    end    

    givenU2 = false
    #println("givenU1 = ", givenU)
    if size(randIntVec2)[1] == 0
        givenU2 = false
    else
        givenU2 = true
    end    

    givenMeasure = false
    #println("givenU1 = ", givenU)
    if size(probArray)[1] == 0
        givenMeasure = false
    else
        givenMeasure = true
    end

    
    if !Bool(withScramble)
        ScrambleLabel = ""
    elseif withScramble
        ScrambleLabel = "Scrambled"
    end 
        
    depth=T #4*L
    #println("L:$L, p:$p, T:$depth, ")

    ancillaVecX = zeros(Int8, 1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)
    #println("ancillaVecX = ", ancillaVecX)
    
    ancillaVecY = zeros(Int8, 1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(Int8, 1, 4*L+3)
    ancillaVecZ[4*L+1] = 1   
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    deterministic = [0, 0, 0]
    deterministicQ = [0, 0, 0]
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    A = [1, 2*Int(2*L)]
    purifyTime = 0

    purifyX = 0
    finalEE = 0
    fixedSign=false    
        GStab = zeros(2*L+1, 2*(2*L+1)+1)
        state=one(Stabilizer, 2*L+1)
        #middleInd = 2*L
        middleInd = L+1 # New version: 08/02/2022
        refInd = 2*L+1
        if withScramble
            for t=1:depth
                for j=1+t%2:2:2*L
                    if Bool(givenU1)                        
                        tempRandU = randu2(fixedSign, t, Int(L), j ÷ 2, randIntVec1[:])
                    else
                        tempRandU = randu2(fixedSign)
                    end
                    apply!(state, tempRandU, [j,j%(2*L)+1])
                end
            end
            proj = project!(state, single_z(2*L+1, middleInd),keep_result=true,phases=true)
            if proj[3]==2
                apply!(state, single_x(2*L+1,middleInd))
            end
            apply!(state, Hadamard, [refInd])
            apply!(state, CNOT, [refInd, middleInd])
        
        else
            apply!(state, Hadamard, [refInd])
            apply!(state, CNOT, [refInd, middleInd])
        end
    
        #println("state after scramble = ", state)
    
        GStab = convBackStab(state)        
        middleEE = Entropy(GStab[1:2*L+1, :], A)
        
        #println("middleEE = ", middleEE)
        
        randArr = zeros(2*L, depth)        
        
        for t=1:depth
            #println("t = ", t)
            for j=1+t%2:2:2*L
                if Bool(givenU2)
                    tempRandU = randu2(fixedSign, t, Int(L), j ÷ 2, randIntVec2[:])
                else
                    tempRandU = randu2(fixedSign)#randu2(fixedSign)
                end
                #println("state tempRandU = ", state)
                apply!(state, tempRandU, [j,j%(2*L)+1])
                GStab = convBackStab(state)

            end
            
            if !givenMeasure
                B = rand(Float64, 2*L)            
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end   
            #println("after scramble = ")            
            for j in 1:2*L 
                if Bool(x[j]) 
                    projection = project!(state, single_z(2*L+1,j),keep_result=keepTraj,phases=true)
                    result = projection[3]
                    if result == 0x0
                        measureRes = 1
                    elseif result == 0x1
                        measureRes = 0.5
                    elseif result == 0x2
                        measureRes = 2
                    elseif result == 0x3
                        measureRes = 1.5  
                    elseif result ==nothing
                        measureRes = rand([0x00, 0x02])
                    end            
                    #println("measureRes = ", result)
                    #    if i == 1.0
                    #        tempI = 1 #S_z = -1
                    #    elseif isnan(i)
                    #        tempI = 0     # No measurement   
                    #    elseif i == 2.0
                    #        tempI = 2  # S_z = +1
                            #print("Sz")
                    #    end
                                    
                    measureResVec[t, j] = measureRes
                    
                    #measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                    #GStab = copy(measurePauliRes[1])                    
                    #measureResVec[t, i] = copy(measurePauliRes[2])
                
                
                    GStab = convBackStab(state)
                    tempEnt = Entropy(GStab[1:2*L+1, :], A)
                    if tempEnt == 0 && purifyTime == 0 && purifyX == 0
                        purifyTime = t
                        purifyX = j
                    end
                    if tempEnt<0
                        println("break negative entropy = ")
                        #break
                    end     
                end
            end
            
            GStab = convBackStab(state)
            
            if t==depth
                finalEE = Entropy(GStab[1:2*L+1, :], A)
                if finalEE<0
                    println("break negative entropy = ")
                end

            end
            
        end
    #println("finalEE = ", finalEE)
    #println("finished time evolution")
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabY)          
    convFinalGStabZ  = convertStabRep(finalGStabZ)          
    
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]        
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]
    #println("measure X")
    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]
    #println("measure Y")
    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    #println("measure Z")    
    deterministic = [deterX, deterY, deterZ]
    #println("determinishtic")
    #println("deterministic = ", deterministic)
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    if purifyTime == 0
        #purifyTime = NaN
    end
    #println("before return")        
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, finalEE     
    #return finalEE, purifyTime
end




function createMeasureRecArg(args) #(L, p, T, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circIndVec=false)
    #job_id=ENV["SLURM_ARRAY_TASK_ID"]
 
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    PT = parse(Int, args[4])
    Scramble = parse(Int, args[5])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    Nsamp = parse(Int, args[6])
    Ncircuit = parse(Int, args[7])
    println("args[8] = ", typeof(args[8]))
    TEvolve = parse(Int, args[8])
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[9]) = ", parse(Int, args[9]))
        Nbatch = parse(Int, args[9])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end
    
    println("Nbatch = ", Nbatch)
    NpureT = parse(Int, args[10])
    NcircPT = parse(Int, args[11])
    circIndVec = parse(Int, args[12])
    
    severalJob = false
    severalJob = parse(Bool, args[13])
    println("severalJOB = ", severalJob)
    if severalJob ==true
        job_id=ENV["SLURM_ARRAY_TASK_ID"]
    end

    csv = parse(Bool, args[14])
    println("csv = ", csv)
    v = false
    try
        v = parse(Int, args[15])
    catch y
        if isa(y, ArgumentError)
            v = false
        end
    end
    
    println(L, p, T, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circIndVec, severalJob, csv, v)
    
    if csv
        println("if csv")
        rv1 = zeros(Ncircuit, L*T);
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv1 = zeros(size(df, 1), size(df, 2));

        println("size rv1 = ", size(rv1), size(rv2))

        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        println("rv1[1, 1:10] = ", rv1[1, 1:10])
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1]
            #println("i = ", i)
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end
    flush(stdout)    
    

    
    @time begin
    #PT = 5;
    #T = 4*L;
    #
    #if circIndVec==false
    #T = PT+1;
    PT = T-1
    #else
    #    T = PT
    #else 
    #p = 0.3;

    probArr = zeros(T, 2*L)
    x = zeros(T, 2*L)
    A = zeros(1, 2*L)    

        
    if p==0.3
        Prob = "0p3"
    elseif p == 0.05
        Prob = "0p05"
            
    elseif p == 0.2
        Prob = "0p2"
    elseif p == 0.1
        Prob = "0p1"
    elseif p == 0.15
        Prob = "0p15"
    elseif p == 0.25
        Prob = "0p25"            
    elseif p == 0.35
        Prob = "0p35"            
    elseif p == 0.4
        Prob = "0p4"            
    end
        
        
    purifyTimeVec = zeros(Ncircuit);
        
    purifyTime = 0
    measureRes = zeros(Nsamp, TEvolve, 2*L+1);
    measureRes[:, :, :] .= NaN;
    finalMeasures = zeros(Nsamp, 3);
    
    determ = zeros(Nsamp, 3);
    fmeasureDeterm = zeros(Nsamp, 3)

    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    probArrMat = zeros(Ncircuit, T, 2*L);
    circuitVec = 1:Ncircuit;
    EEVec = zeros(Ncircuit)
    keepTraj = true
    ## Loading randomVecs and ProbArray of measurements:

    #csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")    
    flush(stdout)
    #println("circIndArr[:, PT] = ", circIndArr[:, PT])
    if circIndVec==false 
        if Nbatch==false
            print("Nbatch==false")            
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                        
        else 
            print("Nbatch==true")            
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")["data"]
            println("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")            
        end    
        circIndVec = circIndArr[1:NcircPT, PT]
    end
    
    for circInd in circIndVec #Int(size(flatCircInd)[1])   
        print("circInd = ", circInd)
        flush(stdout)    
        nb = Int(floor((circInd-1)/Ncircuit))+1

        remainCircInd = (circInd-1)%Ncircuit+1
        if Nbatch==false        
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
        else
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")            
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")        
        end
        
        println("circInd = ", circInd)
        flush(stdout)    

        @inbounds Threads.@threads for i in 1:Nsamp
            if i%1==0
                println("i = $i")
                flush(stdout)   
            end
            #if i%100==0
            #    println("i = $i")
            #end
        
            #A = TimeEvolveAncilla(L, p, T, false, randVec1PureT, randVec2PureT, probArrPureT);    
            #A = QCliffTimeEvolveAncilla(L, p, T, false, randVec1PureT, randVec2PureT, probArrPureT);           
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, TEvolve, Bool(Scramble), rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :]);
            else
                A = QCliffordAncilla(L, p, TEvolve, Bool(Scramble), keepTraj, rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :])            
            end
                
            #println("A[3][:] = ", A[3][:])
            #println("A[4][:] = ", A[4][:])    
            finalMeasures[i, :] = A[3][:]
            determ[i, :] = A[4][:]
            #purifyTimeVec[i] = A[5]

            for j = 1:3
                #println("j = ", j)
                #println("finalMeasxures[i, j] = ", finalMeasures[i, j])
                #println("determ[i, j] = ", determ[i, j])
                fmeasureDeterm[i, j] = finalMeasures[i, j] * determ[i, j]
            end
            #println("end i = ", i)
            #println("A[3] = ", fmeasureDeterm[i, :])
            print("size A2 = ", size(A[2][:, :]   ))
            print("size measureRes = ", size(measureRes))                
            measureRes[i, :, :] = A[2][:, :]   
            purifyTime = A[5]
            println("purifyTime = ", purifyTime)        
        end
        
        swapMeasureRes = permutedims(measureRes, [1, 3, 2])
        #printMatrix(swapMeasureRes[1, :, :])
        if severalJob==false
            if T==4*L && Nbatch==false && v==false
                outfile = "measureL$L\aN$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                outfile = "measureL$L\aNT$T\a$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                    
            elseif T!=4*L && Nbatch==false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                    
            elseif T!=4*L && Nbatch!=false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch==false && v!=false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\av$v$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch!=false && v!=false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch\av$v$ScrambleLabel$ScrambleLabel.txt"
            end
        elseif severalJob==true
            if T==4*L && Nbatch==false && v==false
                outfile = "measureL$L\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"                    
            elseif T!=4*L && Nbatch==false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch!=false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
            end
        end
        
        println("outfile = ", outfile)

        flush(stdout)
    
        open(outfile, "w") do f   
            n = 1    
            #println("n = $n")
            while n <= size(swapMeasureRes)[1]
                sizeI = size(swapMeasureRes[n, :, :])
                #n = 1
                for t in 1:TEvolve
                    tempI = 0
                    for i in swapMeasureRes[n, 1:2*L, t]               

                        if i == 1.0
                            tempI = 1 #S_z = -1
                        elseif isnan(i)
                            tempI = 0     # No measurement           
                        elseif i == 2.0
                            tempI = 2  # S_z = +1
                            #print("Sz")
                        end
                        i = tempI
                        #println("tempI = $tempI")                
                        print(f, i)
                        print(f, ' ')  
                    end
                end
                for i in fmeasureDeterm[n, 1:3] # This prints the final ancilla qbit state to the end of each monitoring 
                    #println("i = ", i)
                    tempI = 0
                    if i == 1.0
                        tempI = 1 #S_z = -1
                    elseif isnan(i)
                        tempI = 0     # No measurement           
                    elseif i == 2.0
                        tempI = 2  # S_z = +1
                    end
                    i = tempI
                    print(f, Int(i))
                    print(f, ' ') 
                end            
                #println(n)        
                #println("purifyTimeVec[1] = ", purifyTimeVec[1])
                #println("purifyTime2 = ", purifyTime)   
                print(f, Int(purifyTime))
                #print(f, Int(purifyTimeVec[remainCircInd]))
                print(f, "\n")  
                n = n + 1
            end
        end
    end
    end
end

#Nsamp = 1;#parse(Int, args[1])
#circInd =2;# parse(thInt, args[2])
#T = 2; #parse(Int, args[3])

#L = 128; p = 0.3; Ncircuit=10; Nbatch=100; NpureT=10; NcircPT=20;
#println(p+circInd)
#A = JuliaCliffordMainModules.createMeasureRec(L, p, T, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circInd)

#A = determinePurifyTimeARGS(ARGS)
#A = circIndArrGenerationARGS(ARGS)


function createMeasureRec(L, p, T, Scramble, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, TEvolve, circIndVec=false, csv=false, v=false)
    if TEvolve==Any[]
        TEvolve = T
    end
    
    println("TEvolve = ", TEvolve)
    #Scramble = parse(Int, args[8])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 


    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    keepTraj = true
    
    if csv
        rv1 = zeros(Ncircuit, L*T);
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv1 = zeros(size(df, 1), size(df, 2));

        #println("size rv1 = ", size(rv1), size(rv2))

        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        #println("rv1[1, 1:10] = ", rv1[1, 1:10])
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1]
            #println("i = ", i)        
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end
    flush(stdout)
    #@time begin;
    #PT = 5;
    #T = 4*L;
    #
    
    if circIndVec==false
        PT = T-1            
    else
        PT = 1
        #continue; 
    end
    #T = PT+1;

    #else
    #    T = PT
    #else 
    
    probArr = zeros(T, 2*L)
    x = zeros(T, 2*L)
    A = zeros(1, 2*L)    
    
    if p==0.3
        Prob = "P0p3"
    elseif p == 0.05
        Prob = "P0p05"            
    elseif p == 0.2
        Prob = "P0p2"
    elseif p == 0.1
        Prob = "P0p1"
    elseif p == 0.15
        Prob = "P0p15"
    elseif p == 0.25
        Prob = "P0p25"            
    elseif p == 0.35
        Prob = "P0p35"            
    elseif p == 0.4
        Prob = "P0p4"            
    end
    
    purifyTimeVec = zeros(Ncircuit);
    measureRes = zeros(Nsamp, TEvolve, 2*L+1);
    measureRes[:, :, :] .= NaN;
    finalMeasures = zeros(Nsamp, 3);
    
    determ = zeros(Nsamp, 3);
    fmeasureDeterm = zeros(Nsamp, 3)

    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    probArrMat = zeros(Ncircuit, T, 2*L);
    circuitVec = 1:Ncircuit;
    EEVec = zeros(Ncircuit)
    
    ## Loading randomVecs and ProbArray of measurements:
    flush(stdout)
    #csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")    
    
    if Nbatch==false || Nbatch==1 
        try             
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                            
        catch y                
            if isa(y, ArgumentError)
                circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                
            end
        end        
    else
        circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")["data"]
    end
    
    print("circIndArr = ", circIndArr)
    #for j in 1:10
    #    for i in 1:20
    #        print(circIndArr[i, j])
    #        print(',')
    #    end
    #    print("\n")
    #end
    #print("PT =", PT)
    #println("circIndArr[:, PT] = ", circIndArr[:, PT])
    
    if circIndVec==false 
        circIndVec = circIndArr[1:NcircPT, PT]
    end
    println("circIndVec in createMeasureRec = ", circIndVec)
    
    for circInd in circIndVec #Int(size(flatCircInd)[1])
        if circInd==0
            continue
        end
        #nb = Int(floor(circInd/Ncircuit))+1
        nb = Int(floor((circInd-1)/Ncircuit))+1
        
        remainCircInd = (circInd-1)%Ncircuit+1

        println("remainCircInd = ", remainCircInd)
        
        if Nbatch == false || Nbatch == 1
            try 
                println("try 1")
                println("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                
                
            catch y                
                if isa(y, ArgumentError)
                    rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
                    rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
                    probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                
                end
            end
        else      
            try
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]                
                println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")                                            
            catch y                
                if isa(y, ArgumentError)                
                    rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                    rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                    probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]                
                    println("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")                            
                end                
            end
        end
        
        println("circInd = ", circInd, " remainCircInd = ", remainCircInd, " nb = ", nb)
        flush(stdout)
        purifyTime = 0
        Threads.@threads for i in 1:Nsamp
            if i%100==0
                println("i = $i")
            end
            #if i%100==0
            #    println("i = $i")
            #end
            flush(stdout)
            #A = TimeEvolveAncilla(L, p, T, false, randVec1PureT, randVec2PureT, probArrPureT);    
            #A = QCliffTimeEvolveAncilla(L, p, T, false, randVec1PureT, randVec2PureT, probArrPureT);   
        
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, TEvolve, Bool(Scramble), rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :]);
            else
                A = QCliffordAncilla(L, p, TEvolve, Bool(Scramble), keepTraj, rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :])            
            end
        
            #println("A[3][:] = ", A[3][:])
            #println("A[4][:] = ", A[4][:])    
            finalMeasures[i, :] = A[3][:]
            determ[i, :] = A[4][:]
            purifyTimeVec[i] = A[5]
            flush(stdout)
            for j = 1:3
                #println("j = ", j)
                #println("finalMeasures[i, j] = ", finalMeasures[i, j])
                #println("determ[i, j] = ", determ[i, j])
                fmeasureDeterm[i, j] = finalMeasures[i, j] * determ[i, j]
            end
            #println("fmeasureDeterm[i, j] = ", fmeasureDeterm[i, :])
            #println("end i = ", i)
            #println("A[2] = ", A[2][:, :])
            #println("A[3] = ", fmeasureDeterm[i, :])
            measureRes[i, :, :] = A[2][:, :]
            purifyTime = A[5]
            if i==1
                println("purifyTime = ", purifyTime)
            end
            flush(stdout)                
        end
        swapMeasureRes = permutedims(measureRes, [1, 3, 2])
        flush(stdout)
        if T==4*L && Nbatch==false
            outfile = "measureL$L\aN$Nsamp\a$Prob\aCrcInd$circInd$ScrambleLabel.txt"
        elseif T!=4*L && Nbatch==false
            outfile = "measureL$L\aT$T\aN$Nsamp\a$Prob\aCrcInd$circInd$ScrambleLabel.txt"
        elseif T!=4*L && Nbatch!=false
            outfile = "measureL$L\aT$T\aN$Nsamp\a$Prob\aCrcInd$circInd\aNbatch$Nbatch$ScrambleLabel.txt"
        end
        println("outfile = ", outfile)
    
        open(outfile, "w") do f   
            n = 1    
            #println("n = $n")
            while n <= size(swapMeasureRes)[1]
                sizeI = size(swapMeasureRes[n, :, :])
                #n = 1
                for t in 1:TEvolve
                    tempI = 0
                    #println("swapMeasureRes[n, 1:2*L, t] = ", swapMeasureRes[n, 1:2*L, t])
                    for i in swapMeasureRes[n, 1:2*L, t]               

                        if i == 1.0
                            tempI = 1 #S_z = -1
                        elseif isnan(i)
                            tempI = 0     # No measurement   
                        elseif i == 2.0
                            tempI = 2  # S_z = +1
                            #print("Sz")
                        end
                        i = tempI
                        #println("tempI = $tempI")
                        print(f, i)
                        print(f, ' ')
                    end
                end
                for k in fmeasureDeterm[n, 1:3] # This prints the final ancilla qbit state to the end of each monitoring 
                    #println("fmeasureDeterm[i, j] = ", fmeasureDeterm[Int(i), :])                        
                    #println("i = ", i)
                    println("fmeasureDeterm[k, j] = ", k)                        
                    tempI = 0
                        
                    if k == 1.0
                        tempI = 1 #S_z = -1
                    elseif isnan(k)
                        tempI = 0     # No measurement           
                    elseif k == 2.0
                        tempI = 2  # S_z = +1
                    end
                    k = tempI
                    println("k, tempI = ", k, tempI)
                    print(f, Int(k))
                    print(f, ' ') 
                end  
                println("end of fmeasureDeterm")
                #println(n)        
                #println("purifyTimeVec[1] = ", purifyTimeVec[1])
                #print(f, Int(purifyTimeVec[remainCircInd]))
                #Threads
                print(f, Int(purifyTime))
                print(f, "\n")  
                flush(stdout)                    
                n = n + 1
            end    
        end
    end
    #end
end

#createMeasureRec(L, p, T, Scramble, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circIndVec=false, csv=false, v=false)


function combinedSampCreator(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT)
    A = randArrGeneration(L, p, T, Ncircuit, Nbatch, Scramble)
    circInd = false;nb=false;csv=false;
    A = determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch, nb, circInd, csv);
    B = circIndArrGeneration(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT);
    #A = determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch=false, nb=false, cVec = false, csv=false)
end



A = createMeasureRecArg(ARGS)
"""
A = randArrGenerationARGS(ARGS)
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    #Nbatch = false
    Nbatch = parse(Int, args[5])    
    
    println("Nbatch = ", Nbatch)
    
    nb = parse(Int, args[6])
    cVec = false
    try
        cVec = parse(Int, args[7])
    catch y
        if isa(y, ArgumentError)
            cVec = false
        endmeasureNewL
    end    
    Ncirc1 = parse(Int, args[8])
    Ncirc2 = parse(Int, args[9])        
    Scramble = parse(Int, args[10])
"""

#export JULIA_NUM_THREADS=16


