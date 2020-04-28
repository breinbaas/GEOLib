Dump file for D-Settlement : Settlement of soil.
==============================================================================
COMPANY    : 

DATE       : 8-3-2019
TIME       : 16:05:20
FILENAME   : D:\DSettlement\Test Results DSettlement\Benchmarks Branch\bm4-7d.sld
CREATED BY : D-Settlement version 19.1.1.23743
==========================    BEGINNING OF DATA     ==========================
[Input Data]
[VERSION]
Soil=1005
Geometry=1000
D-Settlement=1007
[END OF VERSION]

[SOIL COLLECTION]
    1 = number of items
[SOIL]
Sample
SoilColor=16575398
SoilGamDry=14.00
SoilGamWet=14.00
SoilInitialVoidRatio=0.000000
SoilCohesion=0.00
SoilPhi=0.00
SoilPreconIsotacheType=1
SoilPreconKoppejanType=1
SoilUseEquivalentAge=0
SoilEquivalentAge=0.00E+00
SoilPc=1.00E+01
SoilOCR=1.30
SoilPOP=10.00
SoilLimitStress=1.00
SoilDrained=0
SoilApAsApproximationByCpCs=1
SoilCv=6.00E-08
SoilPermeabilityVer=5.000E-02
SoilPermeabilityHorFactor=3.000
SoilStorageType=0
SoilPermeabilityStrainModulus=1.000E+15
SoilUseProbDefaults=1
SoilStdGamDry=0.70
SoilStdGamWet=0.70
SoilStdCv=3.00E-08
SoilStdPc=2.50E+00
SoilStdPriCompIndex=1.500E-02
SoilStdSecCompIndex=5.000E-02
SoilStdSecCompRate=1.250E-02
SoilStdOCR=0.33
SoilStdPermeabilityVer=1.000E-01
SoilStdPOP=2.50
SoilStdPermeabilityHorFactor=0.750
SoilStdInitialVoidRatio=0.000000
SoilStdPermeabilityStrainModulus=0.000E+00
SoilStdLimitStress=0.00
SoilStdCp=9.00E+00
SoilStdCp1=3.00E+00
SoilStdCs=1.80E+01
SoilStdCs1=9.00E+00
SoilStdAp=9.00E+00
SoilStdAsec=9.00E+00
SoilStdCar=0.0000000
SoilStdCa=0.2500000
SoilStdRRatio=0.2500000
SoilStdCRatio=0.2500000
SoilStdSRatio=0.0000000
SoilStdCrIndex=0.2500000
SoilStdCcIndex=0.2500000
SoilStdCswIndex=0.0000000
SoilDistGamDry=2
SoilDistGamWet=2
SoilDistCv=2
SoilDistdPc=2
SoilDistPriCompIndex=2
SoilDistSecCompIndex=2
SoilDistSecCompRate=2
SoilDistOCR=2
SoilDistPermeabilityVer=2
SoilDistPOP=2
SoilDistPermeabilityHorFactor=2
SoilDistInitialVoidRatio=2
SoilDistPermeabilityStrainModulus=2
SoilDistLimitStress=2
SoilDistCp=2
SoilDistCp1=2
SoilDistCs=2
SoilDistCs1=2
SoilDistAp=2
SoilDistAsec=2
SoilDistCar=2
SoilDistCa=2
SoilDistRRatio=2
SoilDistCRatio=2
SoilDistSRatio=2
SoilDistCrIndex=2
SoilDistCcIndex=2
SoilDistCswIndex=2
SoilCorCpCp1=0.01
SoilCorCsCp1=0.01
SoilCorCs1Cp1=0.01
SoilCorApCp1=0.01
SoilCorASecCp1=0.01
SoilCorCrIndexCcIndex=0.01
SoilCorRRatioCRatio=0.01
SoilCorCaCcIndexOrCRatio=0.01
SoilCorPriCompIndexSecCompIndex=0.01
SoilCorSecCompRateSecCompIndex=0.01
SoilCp=3.00E+01
SoilCp1=1.00E+01
SoilCs=6.00E+01
SoilCs1=3.00E+01
SoilAp=3.00E+01
SoilAsec=3.00E+01
SoilCar=0.0000000
SoilCa=1.0000000
SoilCompRatio=0
SoilRRatio=1.0000000
SoilCRatio=1.0000000
SoilSRatio=0.0000000
SoilCrIndex=1.0000000
SoilCcIndex=1.0000000
SoilCswIndex=0.0000000
SoilPriCompIndex=6.000E-02
SoilSecCompIndex=2.000E-01
SoilSecCompRate=5.000E-02
SoilHorizontalBehaviourType=1
SoilElasticity=1.00000E+03
SoilDefaultElasticity=1
[END OF SOIL]
[END OF SOIL COLLECTION]
[GEOMETRY 1D DATA]
1
        0.000
Sample
5
        0.000
       -0.020
        0.000
        0.000
1
1
 0.00000000000000E+0000
 0.00000000000000E+0000
 1.00000000000000E+0000
[END OF GEOMETRY 1D DATA]
[RUN IDENTIFICATION]
Benchmark MSettle bm4-7d
Conversion formulas for a oedometer test
NEN-Koppejan model
[END OF RUN IDENTIFICATION]
[MODEL]
0 : Dimension = 1D
0 : Calculation type = Darcy
0 : Model = NEN - Koppejan
0 : Strain type = Linear
0 : Vertical drains = FALSE
0 : Fit for settlement plate = FALSE
0 : Probabilistic = FALSE
0 : Horizontal displacements = FALSE
0 : Secondary swelling = FALSE
0 : Waspan = FALSE
[END OF MODEL]
[VERTICALS]
    100 = total Mesh
    1 = number of items
       0.000        0.000 = X, Z
[END OF VERTICALS]
[WATER]
        9.81 = Unit Weight of Water
[END OF WATER]
[NON-UNIFORM LOADS]
    0 = number of items
[END OF NON-UNIFORM LOADS]
[WATER LOADS]
    0 = number of items
[END OF WATER LOADS]
[OTHER LOADS]
    9 = number of items
Initial load
3 : Uniform
          -1      1000.00        0.001        0.000 = Time, Gamma, H, Yapplication
Load step 1
3 : Uniform
           0      1000.00        0.001        0.000 = Time, Gamma, H, Yapplication
Load step 2
3 : Uniform
          10      1000.00        0.002        0.000 = Time, Gamma, H, Yapplication
Load step 3
3 : Uniform
          20      1000.00        0.004        0.000 = Time, Gamma, H, Yapplication
Load step 4
3 : Uniform
          30      1000.00        0.008        0.000 = Time, Gamma, H, Yapplication
Load step 5
3 : Uniform
          40      1000.00        0.016        0.000 = Time, Gamma, H, Yapplication
Load step 6
3 : Uniform
          50      1000.00        0.032        0.000 = Time, Gamma, H, Yapplication
Load step 7
3 : Uniform
          60      1000.00        0.064        0.000 = Time, Gamma, H, Yapplication
Load step 8
3 : Uniform
          70      1000.00        0.128        0.000 = Time, Gamma, H, Yapplication
[END OF OTHER LOADS]
[CALCULATION OPTIONS]
5 : Precon. pressure within a layer = Variable, correction at every step
0 : Imaginary surface = FALSE
0 : Submerging = FALSE
0 : Use end time for fit = FALSE
0 : Maintain profile = FALSE
Superelevation
     0 = Time superelevation
    10.00 = Gamma dry superelevation
    10.00 = Gamma wet superelevation
1 : Dispersion conditions layer boundaries top = drained
1 : Dispersion conditions layer boundaries bottom = drained
0 : Stress distribution soil = Buisman
0 : Stress distribution loads = None
 0.10 = Iteration stop criteria submerging [m]
0.000 = Iteration stop criteria submerging minimum layer height [m]
1 = Maximum iteration steps for submerging
 0.10 = Iteration stop criteria desired profile [m]
    1.00 = Load column width imaginary surface [m]
    1.00 = Load column width non-uniform loads [m]
    1.00 = Load column width trapeziform loads [m]
80 = End of consolidation [days]
 1.00000000000000E-0002 = Number of subtime steps
1.000000000E+000 = Reference time
1 : Dissipation = TRUE
       0.000 = X co-ordinate dissipation
0 : Use fit factors = FALSE
       0.000 = X co-ordinate fit
1 : Predict settlements omitting additional loadsteps = TRUE
[END OF CALCULATION OPTIONS]
[RESIDUAL TIMES]
0 : Number of items
[END OF RESIDUAL TIMES]
[FILTER BAND WIDTH]
1 : Number of items
0.05
[END OF FILTER BAND WIDTH]
[PORE PRESSURE METERS]
    0 = number of items
[END OF PORE PRESSURE METERS]
[NON-UNIFORM LOADS PORE PRESSURES]
    0 = number of items
[END OF NON-UNIFORM LOADS PORE PRESSURES]
[OTHER LOADS PORE PRESSURES]
    9 = number of items
Initial load
       0.000 = Top of heightening
Load step 1
       0.000 = Top of heightening
Load step 2
       0.000 = Top of heightening
Load step 3
       0.000 = Top of heightening
Load step 4
       0.000 = Top of heightening
Load step 5
       0.000 = Top of heightening
Load step 6
       0.000 = Top of heightening
Load step 7
       0.000 = Top of heightening
Load step 8
       0.000 = Top of heightening
[END OF OTHER LOADS PORE PRESSURES]
[CALCULATION OPTIONS PORE PRESSURES]
1 : Shear stress = TRUE
1 : calculation method of lateral stress ratio (k0) = Nu
[END OF CALCULATION OPTIONS PORE PRESSURES]
[VERTICAL DRAIN]
0 : Flow type = Radial
       0.000 = Bottom position
       0.000 = Position of the drain pipe
       0.000 = Position of the leftmost drain
       1.000 = Position of the rightmost drain
       1.000 = Center to center distance
       0.100 = Diameter
       0.100 = Width
       0.003 = Thickness
0 = Grid
       0.000 = Begin time
       0.000 = End time
      35.000 = Under pressure for strips and columns
       0.000 = Under pressure for sand wall
       0.000 = Start of drainage
       0.000 = Phreatic level in drain
       0.000 = Water head during dewatering
    10.00 = Tube pressure during dewatering
0 : Flow type = Off
1 = number of items
     0.000       40.000      0.000      0.00 = Time, Under pressure, Water level, Tube pressure
[END OF VERTICAL DRAIN]
[PROBABILISTIC DATA]
Reliability X Co-ordinate=0.000
Residual Settlement=1.00
Maximum Drawings=100
Maximum Iterations=15
Reliability Type=0
Is Reliability Calculation=0
[END OF PROBABILISTIC DATA]
[PROBABILISTIC DEFAULTS]
ProbDefGamDryVar=0.05
ProbDefGamWetVar=0.05
ProbDefPOPVar=0.25
ProbDefOCRVar=0.25
ProbDefPcVar=0.25
ProbDefPermeabilityVerVar=2.50
ProbDefRatioHorVerPermeabilityCvVar=0.25
ProbDefCvVar=0.50
ProbDefCpVar=0.30
ProbDefCp1Var=0.30
ProbDefCsVar=0.30
ProbDefCs1Var=0.30
ProbDefApVar=0.30
ProbDefASecVar=0.30
ProbDefRRCrVar=0.25
ProbDefCRCcVar=0.25
ProbDefCaVar=0.25
ProbDefPriCompIndexVar=0.25
ProbDefSecCompIndexVar=0.25
ProbDefSecCompRateVar=0.25
ProbDefCpCor=0.01
ProbDefCsCor=0.01
ProbDefCs1Cor=0.01
ProbDefApCor=0.01
ProbDefASecCor=0.01
ProbDefRRCrCor=0.01
ProbDefCaCor=0.01
ProbDefPriCompIndexCor=0.01
ProbDefSecCompRateCor=0.01
ProbDefGamDryDist=2
ProbDefGamWetDist=2
ProbDefPOPDist=2
ProbDefOCRDist=2
ProbDefPcDist=2
ProbDefPermeabilityVerDist=2
ProbDefRatioHorVerPermeabilityCvDist=2
ProbDefCvDist=2
ProbDefCpDist=2
ProbDefCp1Dist=2
ProbDefCsDist=2
ProbDefCs1Dist=2
ProbDefApDist=2
ProbDefASecDist=2
ProbDefRRCrDist=2
ProbDefCRCcDist=2
ProbDefCaDist=2
ProbDefPriCompIndexDist=2
ProbDefSecCompIndexDist=2
ProbDefSecCompRateDist=2
ProbDefLayerStd=0.10
ProbDefLayerDist=0
[END OF PROBABILISTIC DEFAULTS]
[FIT OPTIONS]
Fit Maximum Number of Iterations=5
Fit Required Iteration Accuracy=0.0001000000
Fit Required Correlation Coefficient=0.990
[END OF FIT OPTIONS]
[FIT CALCULATION]
Is Fit Calculation=0
Fit Vertical Number=-1
[END OF FIT CALCULATION]
[EPS]
        0.00 = Dry unit weight
        0.00 = Saturated unit weight
        0.00 = Load
        0.00 = Height above surface
[END OF EPS]
[FIT]
    0 = number of items
[END OF FIT]
[End of Input Data]

[Results]
[Verticals Count]
1
[End of Verticals Count]
[Vertical]
      1     =  Vertical count
   0.0000000     =  X co-ordinate
   0.0000000     =  Z co-ordinate
[Time-Settlement per Load]
    121  =  Time step count 
      8  =  Load step count 
   0.0000000      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000   
   0.1000000      0.0004478      0.0004478      0.0004478      0.0004478      0.0004478      0.0004478      0.0004478      0.0004478   
   0.2020378      0.0004658      0.0004658      0.0004658      0.0004658      0.0004658      0.0004658      0.0004658      0.0004658   
   0.3310343      0.0004761      0.0004761      0.0004761      0.0004761      0.0004761      0.0004761      0.0004761      0.0004761   
   0.4941118      0.0004874      0.0004874      0.0004874      0.0004874      0.0004874      0.0004874      0.0004874      0.0004874   
   0.7002747      0.0005000      0.0005000      0.0005000      0.0005000      0.0005000      0.0005000      0.0005000      0.0005000   
   0.9609063      0.0005139      0.0005139      0.0005139      0.0005139      0.0005139      0.0005139      0.0005139      0.0005139   
   1.2903972      0.0005290      0.0005290      0.0005290      0.0005290      0.0005290      0.0005290      0.0005290      0.0005290   
   1.7069403      0.0005453      0.0005453      0.0005453      0.0005453      0.0005453      0.0005453      0.0005453      0.0005453   
   2.2335348      0.0005626      0.0005626      0.0005626      0.0005626      0.0005626      0.0005626      0.0005626      0.0005626   
   2.8992565      0.0005809      0.0005809      0.0005809      0.0005809      0.0005809      0.0005809      0.0005809      0.0005809   
   3.7408633      0.0005999      0.0005999      0.0005999      0.0005999      0.0005999      0.0005999      0.0005999      0.0005999   
   4.8048241      0.0006196      0.0006196      0.0006196      0.0006196      0.0006196      0.0006196      0.0006196      0.0006196   
   6.1498854      0.0006399      0.0006399      0.0006399      0.0006399      0.0006399      0.0006399      0.0006399      0.0006399   
   7.8503144      0.0006607      0.0006607      0.0006607      0.0006607      0.0006607      0.0006607      0.0006607      0.0006607   
  10.0000000      0.0006819      0.0006819      0.0006819      0.0006819      0.0006819      0.0006819      0.0006819      0.0006819   
  10.1000000      0.0006828      0.0011333      0.0011333      0.0011333      0.0011333      0.0011333      0.0011333      0.0011333   
  10.2020378      0.0006837      0.0011560      0.0011560      0.0011560      0.0011560      0.0011560      0.0011560      0.0011560   
  10.3310343      0.0006848      0.0011678      0.0011678      0.0011678      0.0011678      0.0011678      0.0011678      0.0011678   
  10.4941118      0.0006862      0.0011807      0.0011807      0.0011807      0.0011807      0.0011807      0.0011807      0.0011807   
  10.7002747      0.0006879      0.0011952      0.0011952      0.0011952      0.0011952      0.0011952      0.0011952      0.0011952   
  10.9609063      0.0006901      0.0012114      0.0012114      0.0012114      0.0012114      0.0012114      0.0012114      0.0012114   
  11.2903972      0.0006927      0.0012295      0.0012295      0.0012295      0.0012295      0.0012295      0.0012295      0.0012295   
  11.7069403      0.0006959      0.0012492      0.0012492      0.0012492      0.0012492      0.0012492      0.0012492      0.0012492   
  12.2335348      0.0006999      0.0012708      0.0012708      0.0012708      0.0012708      0.0012708      0.0012708      0.0012708   
  12.8992565      0.0007047      0.0012941      0.0012941      0.0012941      0.0012941      0.0012941      0.0012941      0.0012941   
  13.7408633      0.0007104      0.0013191      0.0013191      0.0013191      0.0013191      0.0013191      0.0013191      0.0013191   
  14.8048241      0.0007172      0.0013459      0.0013459      0.0013459      0.0013459      0.0013459      0.0013459      0.0013459   
  16.1498854      0.0007251      0.0013745      0.0013745      0.0013745      0.0013745      0.0013745      0.0013745      0.0013745   
  17.8503144      0.0007344      0.0014048      0.0014048      0.0014048      0.0014048      0.0014048      0.0014048      0.0014048   
  20.0000000      0.0007449      0.0014368      0.0014368      0.0014368      0.0014368      0.0014368      0.0014368      0.0014368   
  20.1000000      0.0007453      0.0014381      0.0018859      0.0018859      0.0018859      0.0018859      0.0018859      0.0018859   
  20.2020378      0.0007458      0.0014395      0.0019146      0.0019146      0.0019146      0.0019146      0.0019146      0.0019146   
  20.3310343      0.0007464      0.0014412      0.0019277      0.0019277      0.0019277      0.0019277      0.0019277      0.0019277   
  20.4941118      0.0007471      0.0014434      0.0019415      0.0019415      0.0019415      0.0019415      0.0019415      0.0019415   
  20.7002747      0.0007481      0.0014461      0.0019570      0.0019570      0.0019570      0.0019570      0.0019570      0.0019570   
  20.9609063      0.0007492      0.0014494      0.0019746      0.0019746      0.0019746      0.0019746      0.0019746      0.0019746   
  21.2903972      0.0007507      0.0014536      0.0019943      0.0019943      0.0019943      0.0019943      0.0019943      0.0019943   
  21.7069403      0.0007525      0.0014587      0.0020160      0.0020160      0.0020160      0.0020160      0.0020160      0.0020160   
  22.2335348      0.0007547      0.0014649      0.0020400      0.0020400      0.0020400      0.0020400      0.0020400      0.0020400   
  22.8992565      0.0007575      0.0014725      0.0020662      0.0020662      0.0020662      0.0020662      0.0020662      0.0020662   
  23.7408633      0.0007608      0.0014817      0.0020949      0.0020949      0.0020949      0.0020949      0.0020949      0.0020949   
  24.8048241      0.0007649      0.0014927      0.0021261      0.0021261      0.0021261      0.0021261      0.0021261      0.0021261   
  26.1498854      0.0007699      0.0015057      0.0021598      0.0021598      0.0021598      0.0021598      0.0021598      0.0021598   
  27.8503144      0.0007758      0.0015209      0.0021964      0.0021964      0.0021964      0.0021964      0.0021964      0.0021964   
  30.0000000      0.0007828      0.0015386      0.0022357      0.0022357      0.0022357      0.0022357      0.0022357      0.0022357   
  30.1000000      0.0007831      0.0015394      0.0022374      0.0032554      0.0032554      0.0032554      0.0032554      0.0032554   
  30.2020378      0.0007834      0.0015402      0.0022391      0.0033484      0.0033484      0.0033484      0.0033484      0.0033484   
  30.3310343      0.0007838      0.0015412      0.0022412      0.0033770      0.0033770      0.0033770      0.0033770      0.0033770   
  30.4941118      0.0007843      0.0015425      0.0022439      0.0034003      0.0034003      0.0034003      0.0034003      0.0034003   
  30.7002747      0.0007850      0.0015440      0.0022472      0.0034257      0.0034257      0.0034257      0.0034257      0.0034257   
  30.9609063      0.0007858      0.0015460      0.0022514      0.0034540      0.0034540      0.0034540      0.0034540      0.0034540   
  31.2903972      0.0007868      0.0015485      0.0022566      0.0034854      0.0034854      0.0034854      0.0034854      0.0034854   
  31.7069403      0.0007880      0.0015516      0.0022630      0.0035200      0.0035200      0.0035200      0.0035200      0.0035200   
  32.2335348      0.0007896      0.0015554      0.0022709      0.0035579      0.0035579      0.0035579      0.0035579      0.0035579   
  32.8992565      0.0007915      0.0015601      0.0022805      0.0035991      0.0035991      0.0035991      0.0035991      0.0035991   
  33.7408633      0.0007939      0.0015659      0.0022921      0.0036437      0.0036437      0.0036437      0.0036437      0.0036437   
  34.8048241      0.0007968      0.0015730      0.0023062      0.0036919      0.0036919      0.0036919      0.0036919      0.0036919   
  36.1498854      0.0008004      0.0015816      0.0023229      0.0037437      0.0037437      0.0037437      0.0037437      0.0037437   
  37.8503144      0.0008048      0.0015920      0.0023427      0.0037995      0.0037995      0.0037995      0.0037995      0.0037995   
  40.0000000      0.0008100      0.0016043      0.0023658      0.0038592      0.0038592      0.0038592      0.0038592      0.0038592   
  40.1000000      0.0008103      0.0016049      0.0023668      0.0038617      0.0050789      0.0050789      0.0050789      0.0050789   
  40.2020378      0.0008105      0.0016055      0.0023679      0.0038643      0.0052367      0.0052367      0.0052367      0.0052367   
  40.3310343      0.0008108      0.0016062      0.0023692      0.0038676      0.0052942      0.0052942      0.0052942      0.0052942   
  40.4941118      0.0008112      0.0016071      0.0023708      0.0038716      0.0053303      0.0053303      0.0053303      0.0053303   
  40.7002747      0.0008117      0.0016082      0.0023729      0.0038767      0.0053634      0.0053634      0.0053634      0.0053634   
  40.9609063      0.0008123      0.0016096      0.0023755      0.0038830      0.0053989      0.0053989      0.0053989      0.0053989   
  41.2903972      0.0008130      0.0016114      0.0023788      0.0038909      0.0054381      0.0054381      0.0054381      0.0054381   
  41.7069403      0.0008140      0.0016136      0.0023828      0.0039005      0.0054815      0.0054815      0.0054815      0.0054815   
  42.2335348      0.0008152      0.0016164      0.0023879      0.0039124      0.0055292      0.0055292      0.0055292      0.0055292   
  42.8992565      0.0008167      0.0016198      0.0023941      0.0039270      0.0055814      0.0055814      0.0055814      0.0055814   
  43.7408633      0.0008185      0.0016241      0.0024019      0.0039446      0.0056384      0.0056384      0.0056384      0.0056384   
  44.8048241      0.0008208      0.0016294      0.0024113      0.0039658      0.0057002      0.0057002      0.0057002      0.0057002   
  46.1498854      0.0008236      0.0016358      0.0024228      0.0039910      0.0057674      0.0057674      0.0057674      0.0057674   
  47.8503144      0.0008271      0.0016437      0.0024368      0.0040209      0.0058400      0.0058400      0.0058400      0.0058400   
  50.0000000      0.0008313      0.0016532      0.0024534      0.0040557      0.0059185      0.0059185      0.0059185      0.0059185   
  50.1000000      0.0008315      0.0016536      0.0024542      0.0040573      0.0059218      0.0068970      0.0068970      0.0068970   
  50.2020378      0.0008317      0.0016541      0.0024550      0.0040589      0.0059253      0.0071623      0.0071623      0.0071623   
  50.3310343      0.0008319      0.0016546      0.0024559      0.0040608      0.0059295      0.0072837      0.0072837      0.0072837   
  50.4941118      0.0008322      0.0016553      0.0024571      0.0040633      0.0059349      0.0073590      0.0073590      0.0073590   
  50.7002747      0.0008326      0.0016562      0.0024587      0.0040665      0.0059416      0.0074130      0.0074130      0.0074130   
  50.9609063      0.0008331      0.0016573      0.0024606      0.0040704      0.0059499      0.0074590      0.0074590      0.0074590   
  51.2903972      0.0008337      0.0016587      0.0024630      0.0040753      0.0059603      0.0075042      0.0075042      0.0075042   
  51.7069403      0.0008345      0.0016604      0.0024660      0.0040815      0.0059731      0.0075522      0.0075522      0.0075522   
  52.2335348      0.0008355      0.0016626      0.0024698      0.0040891      0.0059889      0.0076046      0.0076046      0.0076046   
  52.8992565      0.0008367      0.0016653      0.0024745      0.0040985      0.0060082      0.0076621      0.0076621      0.0076621   
  53.7408633      0.0008382      0.0016687      0.0024803      0.0041102      0.0060316      0.0077253      0.0077253      0.0077253   
  54.8048241      0.0008400      0.0016729      0.0024875      0.0041245      0.0060599      0.0077946      0.0077946      0.0077946   
  56.1498854      0.0008424      0.0016781      0.0024963      0.0041419      0.0060937      0.0078705      0.0078705      0.0078705   
  57.8503144      0.0008452      0.0016845      0.0025072      0.0041629      0.0061336      0.0079536      0.0079536      0.0079536   
  60.0000000      0.0008487      0.0016922      0.0025203      0.0041881      0.0061805      0.0080443      0.0080443      0.0080443   
  60.1000000      0.0008489      0.0016926      0.0025209      0.0041893      0.0061826      0.0080482      0.0086652      0.0086652   
  60.2020378      0.0008490      0.0016929      0.0025215      0.0041904      0.0061847      0.0080521      0.0089945      0.0089945   
  60.3310343      0.0008492      0.0016934      0.0025222      0.0041919      0.0061874      0.0080571      0.0091790      0.0091790   
  60.4941118      0.0008495      0.0016940      0.0025232      0.0041937      0.0061907      0.0080634      0.0093153      0.0093153   
  60.7002747      0.0008498      0.0016947      0.0025244      0.0041960      0.0061950      0.0080712      0.0094245      0.0094245   
  60.9609063      0.0008502      0.0016956      0.0025259      0.0041990      0.0062003      0.0080809      0.0095146      0.0095146   
  61.2903972      0.0008508      0.0016967      0.0025279      0.0042026      0.0062069      0.0080930      0.0095916      0.0095916   
  61.7069403      0.0008514      0.0016982      0.0025303      0.0042072      0.0062152      0.0081080      0.0096607      0.0096607   
  62.2335348      0.0008522      0.0017000      0.0025333      0.0042129      0.0062255      0.0081265      0.0097267      0.0097267   
  62.8992565      0.0008532      0.0017022      0.0025371      0.0042200      0.0062382      0.0081491      0.0097935      0.0097935   
  63.7408633      0.0008545      0.0017050      0.0025418      0.0042288      0.0062540      0.0081767      0.0098641      0.0098641   
  64.8048241      0.0008561      0.0017085      0.0025476      0.0042397      0.0062733      0.0082101      0.0099404      0.0099404   
  66.1498854      0.0008581      0.0017128      0.0025548      0.0042531      0.0062969      0.0082501      0.0100238      0.0100238   
  67.8503144      0.0008605      0.0017181      0.0025636      0.0042695      0.0063255      0.0082977      0.0101153      0.0101153   
  70.0000000      0.0008635      0.0017247      0.0025745      0.0042894      0.0063598      0.0083536      0.0102158      0.0102158   
  70.1000000      0.0008636      0.0017250      0.0025750      0.0042903      0.0063613      0.0083561      0.0102201      0.0105807   
  70.2020378      0.0008638      0.0017253      0.0025755      0.0042912      0.0063629      0.0083587      0.0102246      0.0108363   
  70.3310343      0.0008640      0.0017257      0.0025761      0.0042924      0.0063649      0.0083619      0.0102301      0.0110211   
  70.4941118      0.0008642      0.0017262      0.0025769      0.0042939      0.0063674      0.0083659      0.0102371      0.0111753   
  70.7002747      0.0008645      0.0017268      0.0025779      0.0042957      0.0063706      0.0083710      0.0102457      0.0113126   
  70.9609063      0.0008648      0.0017275      0.0025792      0.0042980      0.0063745      0.0083774      0.0102566      0.0114403   
  71.2903972      0.0008653      0.0017285      0.0025808      0.0043009      0.0063795      0.0083853      0.0102701      0.0115613   
  71.7069403      0.0008658      0.0017297      0.0025828      0.0043046      0.0063857      0.0083953      0.0102868      0.0116766   
  72.2335348      0.0008665      0.0017313      0.0025853      0.0043092      0.0063935      0.0084077      0.0103075      0.0117865   
  72.8992565      0.0008674      0.0017332      0.0025885      0.0043149      0.0064032      0.0084230      0.0103328      0.0118917   
  73.7408633      0.0008685      0.0017356      0.0025924      0.0043220      0.0064152      0.0084420      0.0103638      0.0119936   
  74.8048241      0.0008699      0.0017386      0.0025973      0.0043309      0.0064301      0.0084654      0.0104013      0.0120943   
  76.1498854      0.0008716      0.0017423      0.0026034      0.0043418      0.0064485      0.0084939      0.0104464      0.0121964   
  77.8503144      0.0008737      0.0017469      0.0026109      0.0043553      0.0064709      0.0085286      0.0105002      0.0123027   
  80.0000000      0.0008763      0.0017525      0.0026201      0.0043718      0.0064982      0.0085703      0.0105638      0.0124160   
[End of Time-Settlement per Load]
[Koppejan settlement]
[Column Indication]
Layer number
Primary swelling
Secundary swelling
Primary settlement below pre cons. stress
Secondary settlement below pre cons. stress
Primary settlement above pre cons. stress
Secondary settlement above pre cons. stress
[End of Column Indication]
[Koppejan Settlement Data]
      1  =  Layer seperation count 
      1      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000      0.0000000   
[End of Koppejan Settlement Data]
[End of Koppejan Settlement]
[Depths]
      3    =  Depth count
  -0.0000100
  -0.0100000
  -0.0200000
[End of Depths]
[Leakages]
      2    =  Leakage count
   0.0002669
   0.0002674
[End of Leakages]
[Drained Layers]
      2    =  Layer count
0
0
[End of Drained Layers]
[Time-dependent Data]
[Column Indication]
Settlement
Effective stress vertical
Hydraulic head
Initial Stress
Loading
Upper amplitude convolution
Lower amplitude convolution
Normalized consolidation coefficient
[End of Column Indication]
[Vertical Data at Time]
   0.000000000000    =  Time in days
   0.0000000    1.0000419   -0.0000000    1.0000419    0.0000000    0.0000000    0.0000000       444.2419664 
   0.0000000    1.0419000   -0.0000000    1.0419000    0.0000000    0.0000000    0.0000000       438.0637235 
   0.0000000    1.0838000   -0.0000000    1.0838000    0.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.100000000000    =  Time in days
   0.0004478    2.0000419   -0.0000000    1.0000419    1.0000000    0.1019368    0.1019368       353.9042935 
   0.0002207    1.9978947    0.0044858    1.0419000    1.0000000    0.1019368    0.1019368       351.2984247 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.202037840055    =  Time in days
   0.0004658    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       350.6974104 
   0.0002296    2.0403275    0.0001603    1.0419000    1.0000000    0.0000000    0.0000000       348.1869330 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.331034271074    =  Time in days
   0.0004761    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       348.8767318 
   0.0002347    2.0418548    0.0000046    1.0419000    1.0000000    0.0000000    0.0000000       346.4306971 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.494111803945    =  Time in days
   0.0004874    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       346.8845862 
   0.0002402    2.0418990    0.0000001    1.0419000    1.0000000    0.0000000    0.0000000       344.5093466 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.700274730448    =  Time in days
   0.0005000    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       344.6722531 
   0.0002465    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       342.3752831 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   0.960906293338    =  Time in days
   0.0005139    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       342.2481661 
   0.0002533    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       340.0365100 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   1.290397209704    =  Time in days
   0.0005290    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       339.6280691 
   0.0002608    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       337.5080940 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   1.706940251398    =  Time in days
   0.0005453    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       336.8322042 
   0.0002688    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       334.8094458 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   2.233534755119    =  Time in days
   0.0005626    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       333.8836285 
   0.0002773    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       331.9627024 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   2.899256525960    =  Time in days
   0.0005809    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       330.8065608 
   0.0002863    2.0419000    0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       328.9911365 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   3.740863251897    =  Time in days
   0.0005999    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       327.6249451 
   0.0002957    2.0419000   -0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       325.9177732 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   4.804824071816    =  Time in days
   0.0006196    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       324.3613582 
   0.0003054    2.0419000   -0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       322.7643373 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   6.149885359274    =  Time in days
   0.0006399    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       321.0362945 
   0.0003154    2.0419000   -0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       319.5505618 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
   7.850314391197    =  Time in days
   0.0006607    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       317.6677940 
   0.0003257    2.0419000   -0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       316.2938271 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.000000000000    =  Time in days
   0.0006819    2.0000419   -0.0000000    1.0000419    1.0000000    0.0000000    0.0000000       314.2713398 
   0.0003361    2.0419000   -0.0000000    1.0419000    1.0000000    0.0000000    0.0000000       313.0090602 
   0.0000000    2.0838000   -0.0000000    1.0838000    1.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.100000000000    =  Time in days
   0.0011333    4.0000419   -0.0000000    1.0000419    3.0000000    0.2038736    0.2038736       250.3187655 
   0.0005603    3.9195663    0.0124703    1.0419000    3.0000000    0.2038736    0.2038736       250.1625367 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.202037840055    =  Time in days
   0.0011560    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       247.4798378 
   0.0005715    4.0358290    0.0006189    1.0419000    3.0000000    0.0000000    0.0000000       247.3598582 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.331034271074    =  Time in days
   0.0011678    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       246.0098612 
   0.0005774    4.0416565    0.0000248    1.0419000    3.0000000    0.0000000    0.0000000       245.9138097 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.494111803945    =  Time in days
   0.0011807    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       244.4190200 
   0.0005838    4.0418922    0.0000008    1.0419000    3.0000000    0.0000000    0.0000000       244.3493737 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.700274730448    =  Time in days
   0.0011952    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       242.6341659 
   0.0005910    4.0418998    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       242.5942413 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  10.960906293338    =  Time in days
   0.0012114    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       240.6518081 
   0.0005990    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       240.6450059 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  11.290397209704    =  Time in days
   0.0012295    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       238.4743084 
   0.0006080    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       238.5040434 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  11.706940251398    =  Time in days
   0.0012492    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       236.1063986 
   0.0006178    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       236.1760699 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  12.233534755119    =  Time in days
   0.0012708    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       233.5541829 
   0.0006285    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       233.6671553 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  12.899256525960    =  Time in days
   0.0012941    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       230.8243966 
   0.0006400    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       230.9839877 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  13.740863251897    =  Time in days
   0.0013191    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       227.9240114 
   0.0006524    4.0419000    0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       228.1334754 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  14.804824071816    =  Time in days
   0.0013459    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       224.8602554 
   0.0006657    4.0419000   -0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       225.1227561 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  16.149885359274    =  Time in days
   0.0013745    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       221.6410029 
   0.0006799    4.0419000   -0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       221.9595690 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  17.850314391197    =  Time in days
   0.0014048    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       218.2753893 
   0.0006949    4.0419000   -0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       218.6528501 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.000000000000    =  Time in days
   0.0014368    4.0000419   -0.0000000    1.0000419    3.0000000    0.0000000    0.0000000       214.7744441 
   0.0007107    4.0419000   -0.0000000    1.0419000    3.0000000    0.0000000    0.0000000       215.2133503 
   0.0000000    4.0838000   -0.0000000    1.0838000    3.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.100000000000    =  Time in days
   0.0018859    8.0000419   -0.0000000    1.0000419    7.0000000    0.4077472    0.4077472       171.4098840 
   0.0009346    7.6915251    0.0357161    1.0419000    7.0000000    0.4077472    0.4077472       172.0538539 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.202037840055    =  Time in days
   0.0019146    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       168.9624775 
   0.0009488    8.0171251    0.0025255    1.0419000    7.0000000    0.0000000    0.0000000       169.6142452 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.331034271074    =  Time in days
   0.0019277    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       167.8523587 
   0.0009554    8.0404744    0.0001453    1.0419000    7.0000000    0.0000000    0.0000000       168.5097534 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.494111803945    =  Time in days
   0.0019415    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       166.6946940 
   0.0009622    8.0418339    0.0000067    1.0419000    7.0000000    0.0000000    0.0000000       167.3583987 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.700274730448    =  Time in days
   0.0019570    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       165.3929541 
   0.0009700    8.0418975    0.0000003    1.0419000    7.0000000    0.0000000    0.0000000       166.0639321 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  20.960906293338    =  Time in days
   0.0019746    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       163.9381222 
   0.0009787    8.0418999    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       164.6174397 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  21.290397209704    =  Time in days
   0.0019943    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       162.3278935 
   0.0009885    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       163.0167147 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  21.706940251398    =  Time in days
   0.0020160    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       160.5611798 
   0.0009993    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       161.2607745 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  22.233534755119    =  Time in days
   0.0020400    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       158.6372564 
   0.0010113    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       159.3490075 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  22.899256525960    =  Time in days
   0.0020662    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       156.5553136 
   0.0010243    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       157.2807219 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  23.740863251897    =  Time in days
   0.0020949    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       154.3143335 
   0.0010386    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       155.0550154 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  24.804824071816    =  Time in days
   0.0021261    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       151.9133536 
   0.0010541    8.0419000    0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       152.6710267 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  26.149885359274    =  Time in days
   0.0021598    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       149.3520969 
   0.0010709    8.0419000   -0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       150.1285515 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  27.850314391197    =  Time in days
   0.0021964    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       146.6318656 
   0.0010890    8.0419000   -0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       147.4289178 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.000000000000    =  Time in days
   0.0022357    8.0000419   -0.0000000    1.0000419    7.0000000    0.0000000    0.0000000       143.7565225 
   0.0011086    8.0419000   -0.0000000    1.0419000    7.0000000    0.0000000    0.0000000       144.5759508 
   0.0000000    8.0838000   -0.0000000    1.0838000    7.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.100000000000    =  Time in days
   0.0032554   16.0000419   -0.0000000    1.0000419   15.0000000    0.8154944    0.8154944        86.2366178 
   0.0016178   15.0260200    0.1035556    1.0419000   15.0000000    0.8154944    0.8154944        86.8822651 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.202037840055    =  Time in days
   0.0033484   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        82.3112549 
   0.0016642   15.9086873    0.0135793    1.0419000   15.0000000    0.0000000    0.0000000        82.9438149 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.331034271074    =  Time in days
   0.0033770   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        81.1372947 
   0.0016785   16.0270886    0.0015098    1.0419000   15.0000000    0.0000000    0.0000000        81.7661387 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.494111803945    =  Time in days
   0.0034003   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        80.1944958 
   0.0016901   16.0405489    0.0001377    1.0419000   15.0000000    0.0000000    0.0000000        80.8205329 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.700274730448    =  Time in days
   0.0034257   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        79.1823694 
   0.0017028   16.0417995    0.0000102    1.0419000   15.0000000    0.0000000    0.0000000        79.8054904 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  30.960906293338    =  Time in days
   0.0034540   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        78.0663957 
   0.0017169   16.0418939    0.0000006    1.0419000   15.0000000    0.0000000    0.0000000        78.6864014 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  31.290397209704    =  Time in days
   0.0034854   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        76.8454103 
   0.0017326   16.0418997    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        77.4621372 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  31.706940251398    =  Time in days
   0.0035200   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        75.5224034 
   0.0017498   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        76.1357453 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  32.233534755119    =  Time in days
   0.0035579   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        74.1010661 
   0.0017687   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        74.7109822 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  32.899256525960    =  Time in days
   0.0035991   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        72.5850710 
   0.0017893   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        73.1915891 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  33.740863251897    =  Time in days
   0.0036437   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        70.9777851 
   0.0018115   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        71.5810010 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  34.804824071816    =  Time in days
   0.0036919   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        69.2822719 
   0.0018356   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        69.8823440 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  36.149885359274    =  Time in days
   0.0037437   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        67.5015610 
   0.0018614   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        68.0987003 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  37.850314391197    =  Time in days
   0.0037995   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        65.6391245 
   0.0018892   16.0419000    0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        66.2335788 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.000000000000    =  Time in days
   0.0038592   16.0000419   -0.0000000    1.0000419   15.0000000    0.0000000    0.0000000        63.6994592 
   0.0019189   16.0419000   -0.0000000    1.0419000   15.0000000    0.0000000    0.0000000        64.2914932 
   0.0000000   16.0838000   -0.0000000    1.0838000   15.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.100000000000    =  Time in days
   0.0050789   32.0000419   -0.0000000    1.0000419   31.0000000    1.6309888    1.6309888        34.5895644 
   0.0025286   27.9034803    0.4218573    1.0419000   31.0000000    1.6309888    1.6309888        34.9450780 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.202037840055    =  Time in days
   0.0052367   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        31.9591004 
   0.0026074   30.9113349    0.1152462    1.0419000   31.0000000    0.0000000    0.0000000        32.2973419 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.331034271074    =  Time in days
   0.0052942   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        31.0501630 
   0.0026361   31.7667589    0.0280470    1.0419000   31.0000000    0.0000000    0.0000000        31.3816739 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.494111803945    =  Time in days
   0.0053303   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        30.4950735 
   0.0026541   31.9847919    0.0058214    1.0419000   31.0000000    0.0000000    0.0000000        30.8222809 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.700274730448    =  Time in days
   0.0053634   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        29.9934712 
   0.0026707   32.0319517    0.0010141    1.0419000   31.0000000    0.0000000    0.0000000        30.3167813 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  40.960906293338    =  Time in days
   0.0053989   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        29.4646053 
   0.0026884   32.0404571    0.0001471    1.0419000   31.0000000    0.0000000    0.0000000        29.7838245 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  41.290397209704    =  Time in days
   0.0054381   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        28.8905725 
   0.0027080   32.0417266    0.0000177    1.0419000   31.0000000    0.0000000    0.0000000        29.2053946 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  41.706940251398    =  Time in days
   0.0054815   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        28.2691237 
   0.0027297   32.0418828    0.0000018    1.0419000   31.0000000    0.0000000    0.0000000        28.5792430 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  42.233534755119    =  Time in days
   0.0055292   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        27.6012463 
   0.0027535   32.0418986    0.0000001    1.0419000   31.0000000    0.0000000    0.0000000        27.9063830 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  42.899256525960    =  Time in days
   0.0055814   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        26.8884339 
   0.0027796   32.0418999    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        27.1883392 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  43.740863251897    =  Time in days
   0.0056384   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        26.1321906 
   0.0028080   32.0419000    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        26.4266462 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  44.804824071816    =  Time in days
   0.0057002   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        25.3340370 
   0.0028389   32.0419000    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        25.6228532 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  46.149885359274    =  Time in days
   0.0057674   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        24.4956881 
   0.0028724   32.0419000    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        24.7787016 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  47.850314391197    =  Time in days
   0.0058400   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        23.6193192 
   0.0029086   32.0419000    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        23.8963905 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.000000000000    =  Time in days
   0.0059185   32.0000419   -0.0000000    1.0000419   31.0000000    0.0000000    0.0000000        22.7078558 
   0.0029478   32.0419000    0.0000000    1.0419000   31.0000000    0.0000000    0.0000000        22.9788668 
   0.0000000   32.0838000   -0.0000000    1.0838000   31.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.100000000000    =  Time in days
   0.0068970   64.0000419   -0.0000000    1.0000419   63.0000000    3.2619776    3.2619776        13.9200498 
   0.0034374   46.3855938    1.7998273    1.0419000   63.0000000    3.2619776    3.2619776        14.0825065 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.202037840055    =  Time in days
   0.0071623   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        12.1859341 
   0.0035699   55.4170289    0.8791918    1.0419000   63.0000000    0.0000000    0.0000000        12.3359040 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.331034271074    =  Time in days
   0.0072837   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        11.4670271 
   0.0036305   60.0737205    0.4045035    1.0419000   63.0000000    0.0000000    0.0000000        11.6103715 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.494111803945    =  Time in days
   0.0073590   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        11.0423100 
   0.0036681   62.3895724    0.1684330    1.0419000   63.0000000    0.0000000    0.0000000        11.1815301 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.700274730448    =  Time in days
   0.0074130   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        10.7478579 
   0.0036951   63.4329654    0.0620728    1.0419000   63.0000000    0.0000000    0.0000000        10.8840980 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  50.960906293338    =  Time in days
   0.0074590   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        10.5032634 
   0.0037180   63.8462100    0.0199480    1.0419000   63.0000000    0.0000000    0.0000000        10.6369381 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  51.290397209704    =  Time in days
   0.0075042   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        10.2681162 
   0.0037406   63.9877116    0.0055238    1.0419000   63.0000000    0.0000000    0.0000000        10.3993086 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  51.706940251398    =  Time in days
   0.0075522   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000        10.0242369 
   0.0037646   64.0290839    0.0013064    1.0419000   63.0000000    0.0000000    0.0000000        10.1528624 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  52.233534755119    =  Time in days
   0.0076046   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         9.7646961 
   0.0037908   64.0393287    0.0002621    1.0419000   63.0000000    0.0000000    0.0000000         9.8906085 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  52.899256525960    =  Time in days
   0.0076621   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         9.4873646 
   0.0038195   64.0414647    0.0000444    1.0419000   63.0000000    0.0000000    0.0000000         9.6104046 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  53.740863251897    =  Time in days
   0.0077253   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         9.1918005 
   0.0038511   64.0418381    0.0000063    1.0419000   63.0000000    0.0000000    0.0000000         9.3118115 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  54.804824071816    =  Time in days
   0.0077946   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         8.8781080 
   0.0038857   64.0418926    0.0000008    1.0419000   63.0000000    0.0000000    0.0000000         8.9949401 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  56.149885359274    =  Time in days
   0.0078705   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         8.5466775 
   0.0039236   64.0418993    0.0000001    1.0419000   63.0000000    0.0000000    0.0000000         8.6601894 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  57.850314391197    =  Time in days
   0.0079536   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         8.1982351 
   0.0039651   64.0418999    0.0000000    1.0419000   63.0000000    0.0000000    0.0000000         8.3082951 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.000000000000    =  Time in days
   0.0080443   64.0000419   -0.0000000    1.0000419   63.0000000    0.0000000    0.0000000         7.8339684 
   0.0040104   64.0419000    0.0000000    1.0419000   63.0000000    0.0000000    0.0000000         7.9404567 
   0.0000000   64.0838000   -0.0000000    1.0838000   63.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.100000000000    =  Time in days
   0.0086652  128.0000419   -0.0000000    1.0000419  127.0000000    6.5239551    6.5239551         5.7456840 
   0.0043216   71.6225176    5.7512113    1.0419000  127.0000000    6.5239551    6.5239551         5.8167529 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.202037840055    =  Time in days
   0.0089945  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         4.8716826 
   0.0044861   86.6215757    4.2222553    1.0419000  127.0000000    0.0000000    0.0000000         4.9344169 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.331034271074    =  Time in days
   0.0091790  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         4.4411076 
   0.0045782   99.5867773    2.9006241    1.0419000  127.0000000    0.0000000    0.0000000         4.5005307 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.494111803945    =  Time in days
   0.0093153  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         4.1478900 
   0.0046463  109.3889790    1.9014191    1.0419000  127.0000000    0.0000000    0.0000000         4.2043266 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.700274730448    =  Time in days
   0.0094245  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.9271347 
   0.0047008  116.5315612    1.1733271    1.0419000  127.0000000    0.0000000    0.0000000         3.9811100 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  60.960906293338    =  Time in days
   0.0095146  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.7538604 
   0.0047458  121.4553312    0.6714137    1.0419000  127.0000000    0.0000000    0.0000000         3.8059299 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  61.290397209704    =  Time in days
   0.0095916  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.6119571 
   0.0047842  124.5975182    0.3511093    1.0419000  127.0000000    0.0000000    0.0000000         3.6623949 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  61.706940251398    =  Time in days
   0.0096607  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.4890980 
   0.0048188  126.4163730    0.1657010    1.0419000  127.0000000    0.0000000    0.0000000         3.5380843 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  62.233534755119    =  Time in days
   0.0097267  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.3756305 
   0.0048518  127.3578185    0.0697331    1.0419000  127.0000000    0.0000000    0.0000000         3.4232504 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  62.899256525960    =  Time in days
   0.0097935  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.2645199 
   0.0048852  127.7876939    0.0259130    1.0419000  127.0000000    0.0000000    0.0000000         3.3107924 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  63.740863251897    =  Time in days
   0.0098641  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.1511889 
   0.0049204  127.9591630    0.0084339    1.0419000  127.0000000    0.0000000    0.0000000         3.1960870 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  64.804824071816    =  Time in days
   0.0099404  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         3.0330749 
   0.0049586  128.0184720    0.0023882    1.0419000  127.0000000    0.0000000    0.0000000         3.0765473 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  66.149885359274    =  Time in days
   0.0100238  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         2.9090037 
   0.0050002  128.0361599    0.0005851    1.0419000  127.0000000    0.0000000    0.0000000         2.9509884 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  67.850314391197    =  Time in days
   0.0101153  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         2.7786496 
   0.0050460  128.0406881    0.0001235    1.0419000  127.0000000    0.0000000    0.0000000         2.8190820 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.000000000000    =  Time in days
   0.0102158  128.0000419   -0.0000000    1.0000419  127.0000000    0.0000000    0.0000000         2.6422104 
   0.0050962  128.0416802    0.0000224    1.0419000  127.0000000    0.0000000    0.0000000         2.6810287 
   0.0000000  128.0838000   -0.0000000    1.0838000  127.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.100000000000    =  Time in days
   0.0105807  256.0000419   -0.0000000    1.0000419  255.0000000   13.0479103   13.0479103         2.2025607 
   0.0052793  128.6413383   12.9868055    1.0419000  255.0000000   13.0479103   13.0479103         2.2324532 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.202037840055    =  Time in days
   0.0108363  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.9386100 
   0.0054073  135.5816486   12.2793325    1.0419000  255.0000000    0.0000000    0.0000000         1.9640736 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.331034271074    =  Time in days
   0.0110211  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.7672004 
   0.0054997  150.5145500   10.7571203    1.0419000  255.0000000    0.0000000    0.0000000         1.7908323 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.494111803945    =  Time in days
   0.0111753  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.6356860 
   0.0055766  168.1370467    8.9607394    1.0419000  255.0000000    0.0000000    0.0000000         1.6581822 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.700274730448    =  Time in days
   0.0113126  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.5268155 
   0.0056451  185.3465809    7.2064546    1.0419000  255.0000000    0.0000000    0.0000000         1.5484477 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  70.960906293338    =  Time in days
   0.0114403  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.4322572 
   0.0057089  201.0525917    5.6054341    1.0419000  255.0000000    0.0000000    0.0000000         1.4527622 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  71.290397209704    =  Time in days
   0.0115613  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.3479911 
   0.0057694  214.8800610    4.1959061    1.0419000  255.0000000    0.0000000    0.0000000         1.3675350 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  71.706940251398    =  Time in days
   0.0116766  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.2723736 
   0.0058270  226.6315341    2.9979986    1.0419000  255.0000000    0.0000000    0.0000000         1.2909704 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  72.233534755119    =  Time in days
   0.0117865  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.2042379 
   0.0058819  236.1627628    2.0264156    1.0419000  255.0000000    0.0000000    0.0000000         1.2219981 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  72.899256525960    =  Time in days
   0.0118917  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.1424371 
   0.0059344  243.4445695    1.2841315    1.0419000  255.0000000    0.0000000    0.0000000         1.1594347 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  73.740863251897    =  Time in days
   0.0119936  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.0855902 
   0.0059854  248.6336061    0.7551778    1.0419000  255.0000000    0.0000000    0.0000000         1.1018664 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  74.804824071816    =  Time in days
   0.0120943  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         1.0322098 
   0.0060357  252.0352584    0.4084242    1.0419000  255.0000000    0.0000000    0.0000000         1.0477990 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  76.149885359274    =  Time in days
   0.0121964  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         0.9807619 
   0.0060867  254.0681103    0.2012018    1.0419000  255.0000000    0.0000000    0.0000000         0.9956800 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  77.850314391197    =  Time in days
   0.0123027  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         0.9299009 
   0.0061398  255.1632692    0.0895648    1.0419000  255.0000000    0.0000000    0.0000000         0.9441522 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[Vertical Data at Time]
  80.000000000000    =  Time in days
   0.0124160  256.0000419   -0.0000000    1.0000419  255.0000000    0.0000000    0.0000000         0.0000000 
   0.0061964  255.6907255    0.0357976    1.0419000  255.0000000    0.0000000    0.0000000         0.0000000 
   0.0000000  256.0838000   -0.0000000    1.0838000  255.0000000 -1 -1 -1
[End of Vertical Data at Time]
[End of Time dependent Data]
[Time-dependent Data]
[Column Indication]
Depth
Settlement
[End of Column Indication]
[Vertical Data at Fixed Time]
   0.001000000000    =  Time in days
  -0.0000100    0.0000045
  -0.0010095    0.0000043
  -0.0020090    0.0000040
  -0.0030085    0.0000038
  -0.0040080    0.0000036
  -0.0050075    0.0000033
  -0.0060070    0.0000031
  -0.0070065    0.0000029
  -0.0080060    0.0000027
  -0.0090055    0.0000024
  -0.0100050    0.0000022
  -0.0110045    0.0000020
  -0.0120040    0.0000018
  -0.0130035    0.0000015
  -0.0140030    0.0000013
  -0.0150025    0.0000011
  -0.0160020    0.0000009
  -0.0170015    0.0000007
  -0.0180010    0.0000004
  -0.0190005    0.0000002
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.002000000000    =  Time in days
  -0.0000100    0.0000090
  -0.0010095    0.0000085
  -0.0020090    0.0000080
  -0.0030085    0.0000076
  -0.0040080    0.0000071
  -0.0050075    0.0000067
  -0.0060070    0.0000062
  -0.0070065    0.0000058
  -0.0080060    0.0000053
  -0.0090055    0.0000049
  -0.0100050    0.0000044
  -0.0110045    0.0000040
  -0.0120040    0.0000035
  -0.0130035    0.0000031
  -0.0140030    0.0000026
  -0.0150025    0.0000022
  -0.0160020    0.0000018
  -0.0170015    0.0000013
  -0.0180010    0.0000009
  -0.0190005    0.0000004
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.005000000000    =  Time in days
  -0.0000100    0.0000224
  -0.0010095    0.0000213
  -0.0020090    0.0000201
  -0.0030085    0.0000190
  -0.0040080    0.0000178
  -0.0050075    0.0000167
  -0.0060070    0.0000156
  -0.0070065    0.0000144
  -0.0080060    0.0000133
  -0.0090055    0.0000122
  -0.0100050    0.0000110
  -0.0110045    0.0000099
  -0.0120040    0.0000088
  -0.0130035    0.0000077
  -0.0140030    0.0000066
  -0.0150025    0.0000055
  -0.0160020    0.0000044
  -0.0170015    0.0000033
  -0.0180010    0.0000022
  -0.0190005    0.0000011
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.010000000000    =  Time in days
  -0.0000100    0.0000448
  -0.0010095    0.0000425
  -0.0020090    0.0000402
  -0.0030085    0.0000380
  -0.0040080    0.0000357
  -0.0050075    0.0000334
  -0.0060070    0.0000312
  -0.0070065    0.0000289
  -0.0080060    0.0000266
  -0.0090055    0.0000243
  -0.0100050    0.0000221
  -0.0110045    0.0000199
  -0.0120040    0.0000176
  -0.0130035    0.0000154
  -0.0140030    0.0000132
  -0.0150025    0.0000110
  -0.0160020    0.0000088
  -0.0170015    0.0000066
  -0.0180010    0.0000044
  -0.0190005    0.0000022
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.020000000000    =  Time in days
  -0.0000100    0.0000896
  -0.0010095    0.0000850
  -0.0020090    0.0000805
  -0.0030085    0.0000759
  -0.0040080    0.0000714
  -0.0050075    0.0000668
  -0.0060070    0.0000623
  -0.0070065    0.0000578
  -0.0080060    0.0000532
  -0.0090055    0.0000487
  -0.0100050    0.0000441
  -0.0110045    0.0000397
  -0.0120040    0.0000353
  -0.0130035    0.0000309
  -0.0140030    0.0000265
  -0.0150025    0.0000221
  -0.0160020    0.0000176
  -0.0170015    0.0000132
  -0.0180010    0.0000088
  -0.0190005    0.0000044
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.050000000000    =  Time in days
  -0.0000100    0.0002239
  -0.0010095    0.0002126
  -0.0020090    0.0002012
  -0.0030085    0.0001898
  -0.0040080    0.0001785
  -0.0050075    0.0001671
  -0.0060070    0.0001558
  -0.0070065    0.0001444
  -0.0080060    0.0001330
  -0.0090055    0.0001217
  -0.0100050    0.0001103
  -0.0110045    0.0000993
  -0.0120040    0.0000882
  -0.0130035    0.0000772
  -0.0140030    0.0000662
  -0.0150025    0.0000552
  -0.0160020    0.0000441
  -0.0170015    0.0000331
  -0.0180010    0.0000221
  -0.0190005    0.0000110
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.100000000000    =  Time in days
  -0.0000100    0.0004478
  -0.0010095    0.0004251
  -0.0020090    0.0004024
  -0.0030085    0.0003797
  -0.0040080    0.0003570
  -0.0050075    0.0003342
  -0.0060070    0.0003115
  -0.0070065    0.0002888
  -0.0080060    0.0002661
  -0.0090055    0.0002433
  -0.0100050    0.0002206
  -0.0110045    0.0001986
  -0.0120040    0.0001765
  -0.0130035    0.0001544
  -0.0140030    0.0001324
  -0.0150025    0.0001103
  -0.0160020    0.0000882
  -0.0170015    0.0000662
  -0.0180010    0.0000441
  -0.0190005    0.0000221
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.200000000000    =  Time in days
  -0.0000100    0.0004656
  -0.0010095    0.0004420
  -0.0020090    0.0004183
  -0.0030085    0.0003947
  -0.0040080    0.0003711
  -0.0050075    0.0003475
  -0.0060070    0.0003239
  -0.0070065    0.0003002
  -0.0080060    0.0002766
  -0.0090055    0.0002530
  -0.0100050    0.0002294
  -0.0110045    0.0002064
  -0.0120040    0.0001835
  -0.0130035    0.0001606
  -0.0140030    0.0001376
  -0.0150025    0.0001147
  -0.0160020    0.0000918
  -0.0170015    0.0000688
  -0.0180010    0.0000459
  -0.0190005    0.0000229
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   0.500000000000    =  Time in days
  -0.0000100    0.0004878
  -0.0010095    0.0004631
  -0.0020090    0.0004383
  -0.0030085    0.0004136
  -0.0040080    0.0003888
  -0.0050075    0.0003641
  -0.0060070    0.0003393
  -0.0070065    0.0003146
  -0.0080060    0.0002898
  -0.0090055    0.0002651
  -0.0100050    0.0002403
  -0.0110045    0.0002163
  -0.0120040    0.0001923
  -0.0130035    0.0001682
  -0.0140030    0.0001442
  -0.0150025    0.0001202
  -0.0160020    0.0000961
  -0.0170015    0.0000721
  -0.0180010    0.0000481
  -0.0190005    0.0000240
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   1.000000000000    =  Time in days
  -0.0000100    0.0005159
  -0.0010095    0.0004898
  -0.0020090    0.0004636
  -0.0030085    0.0004374
  -0.0040080    0.0004112
  -0.0050075    0.0003851
  -0.0060070    0.0003589
  -0.0070065    0.0003327
  -0.0080060    0.0003065
  -0.0090055    0.0002804
  -0.0100050    0.0002542
  -0.0110045    0.0002288
  -0.0120040    0.0002034
  -0.0130035    0.0001779
  -0.0140030    0.0001525
  -0.0150025    0.0001271
  -0.0160020    0.0001017
  -0.0170015    0.0000763
  -0.0180010    0.0000508
  -0.0190005    0.0000254
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   2.000000000000    =  Time in days
  -0.0000100    0.0005555
  -0.0010095    0.0005273
  -0.0020090    0.0004992
  -0.0030085    0.0004710
  -0.0040080    0.0004428
  -0.0050075    0.0004146
  -0.0060070    0.0003864
  -0.0070065    0.0003582
  -0.0080060    0.0003301
  -0.0090055    0.0003019
  -0.0100050    0.0002737
  -0.0110045    0.0002463
  -0.0120040    0.0002190
  -0.0130035    0.0001916
  -0.0140030    0.0001642
  -0.0150025    0.0001368
  -0.0160020    0.0001095
  -0.0170015    0.0000821
  -0.0180010    0.0000547
  -0.0190005    0.0000274
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
   5.000000000000    =  Time in days
  -0.0000100    0.0006229
  -0.0010095    0.0005913
  -0.0020090    0.0005597
  -0.0030085    0.0005281
  -0.0040080    0.0004965
  -0.0050075    0.0004649
  -0.0060070    0.0004333
  -0.0070065    0.0004017
  -0.0080060    0.0003701
  -0.0090055    0.0003385
  -0.0100050    0.0003069
  -0.0110045    0.0002762
  -0.0120040    0.0002455
  -0.0130035    0.0002148
  -0.0140030    0.0001841
  -0.0150025    0.0001535
  -0.0160020    0.0001228
  -0.0170015    0.0000921
  -0.0180010    0.0000614
  -0.0190005    0.0000307
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
  10.000000000000    =  Time in days
  -0.0000100    0.0006819
  -0.0010095    0.0006473
  -0.0020090    0.0006127
  -0.0030085    0.0005781
  -0.0040080    0.0005435
  -0.0050075    0.0005089
  -0.0060070    0.0004743
  -0.0070065    0.0004397
  -0.0080060    0.0004051
  -0.0090055    0.0003706
  -0.0100050    0.0003360
  -0.0110045    0.0003024
  -0.0120040    0.0002688
  -0.0130035    0.0002352
  -0.0140030    0.0002016
  -0.0150025    0.0001680
  -0.0160020    0.0001344
  -0.0170015    0.0001008
  -0.0180010    0.0000672
  -0.0190005    0.0000336
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
  20.000000000000    =  Time in days
  -0.0000100    0.0014368
  -0.0010095    0.0013641
  -0.0020090    0.0012915
  -0.0030085    0.0012189
  -0.0040080    0.0011462
  -0.0050075    0.0010736
  -0.0060070    0.0010009
  -0.0070065    0.0009283
  -0.0080060    0.0008557
  -0.0090055    0.0007830
  -0.0100050    0.0007104
  -0.0110045    0.0006393
  -0.0120040    0.0005683
  -0.0130035    0.0004973
  -0.0140030    0.0004262
  -0.0150025    0.0003552
  -0.0160020    0.0002842
  -0.0170015    0.0002131
  -0.0180010    0.0001421
  -0.0190005    0.0000710
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
  50.000000000000    =  Time in days
  -0.0000100    0.0059185
  -0.0010095    0.0056213
  -0.0020090    0.0053240
  -0.0030085    0.0050268
  -0.0040080    0.0047296
  -0.0050075    0.0044324
  -0.0060070    0.0041352
  -0.0070065    0.0038380
  -0.0080060    0.0035407
  -0.0090055    0.0032435
  -0.0100050    0.0029463
  -0.0110045    0.0026517
  -0.0120040    0.0023571
  -0.0130035    0.0020624
  -0.0140030    0.0017678
  -0.0150025    0.0014732
  -0.0160020    0.0011785
  -0.0170015    0.0008839
  -0.0180010    0.0005893
  -0.0190005    0.0002946
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
 100.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
 200.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
 500.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
1000.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
2000.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
5000.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[Vertical Data at Fixed Time]
10000.000000000000    =  Time in days
  -0.0000100    0.0124160
  -0.0010095    0.0117937
  -0.0020090    0.0111714
  -0.0030085    0.0105492
  -0.0040080    0.0099269
  -0.0050075    0.0093046
  -0.0060070    0.0086824
  -0.0070065    0.0080601
  -0.0080060    0.0074379
  -0.0090055    0.0068156
  -0.0100050    0.0061933
  -0.0110045    0.0055740
  -0.0120040    0.0049547
  -0.0130035    0.0043353
  -0.0140030    0.0037160
  -0.0150025    0.0030967
  -0.0160020    0.0024773
  -0.0170015    0.0018580
  -0.0180010    0.0012387
  -0.0190005    0.0006193
  -0.0200000    0.0000000
[End of Vertical Data at Fixed Time]
[End of Time dependent Data]
[End of Vertical]
[RESIDUAL SETTLEMENTS]
[Column Indication]
Vertical
Time
Settlement
Part of final settlement
Residual settlement
[End of Column Indication]
[DATA COUNT]
      0 Data count
[END OF DATA COUNT]
[RESIDUAL SETTLEMENT DATA]
[END OF RESIDUAL SETTLEMENT DATA]
[END OF RESIDUAL SETTLEMENTS]
[Dissipation in Layers]
      1  =  Number of Layers 
     25  =  Number of time steps 
[Dissipation layer]
      1  =  Layer Number 
   0.0000000      0.0000000   
   0.1000000      0.9902590   
   0.2008680      0.9942116   
   0.3280799      0.9964640   
   0.4885161      0.9979693   
   0.6908539      0.9989033   
   0.9460369      0.9994418   
   1.2678666      0.9997316   
   1.6737496      0.9998777   
   2.1856381      0.9999470   
   2.8312180      0.9999781   
   3.6454059      0.9999913   
   4.6722374      0.9999967   
   5.9672493      0.9999988   
   7.6004832      0.9999996   
   9.6602733      0.9999999   
  12.2580245      1.0000000   
  15.5342377      1.0000000   
  19.6661086      1.0000000   
  24.8771118      1.0000000   
  31.4490873      1.0000000   
  39.7374840      1.0000000   
  50.1905843      1.0000000   
  63.3737500      1.0000000   
  80.0000000      1.0000000   
[End of Dissipation layer]
[End of Dissipation in Layers]
[RESIDUAL SETTLEMENTS]
[Column Indication]
Vertical
Time
Settlement
Part of final settlement
Residual settlement
[End of Column Indication]
[DATA COUNT]
      0 Data count
[END OF DATA COUNT]
[RESIDUAL SETTLEMENT DATA]
[END OF RESIDUAL SETTLEMENT DATA]
[END OF RESIDUAL SETTLEMENTS]
[End of Results]
