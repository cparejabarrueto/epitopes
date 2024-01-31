library(devtools)
#install_github("dosorio/Peptides")
#install.packages("tidyverse")
#install.packages("tibble")
library(Peptides)

#Hacer CSV con datos de epitopes
#Desde la descarga de IEDB leer la columna de las secuencias
#Revisar los aminoácidos modificados
#Leer CSV con los datos de los epitopes
#csv_initial = read.csv('epitope_table_export_256.csv',sep = ";")
csv_initial = read.csv('input_script.txt',sep = ";",header = F)
#csv_initial = read.csv('../Epitopes/Test/covid.txt',sep = ";",header = F)
seqs <- csv_initial$V1
seqs[1]
rows <- nrow(csv_initial)
#Crear CSV nuevo
#csv<-data.frame(matrix(NA,nrow=256,ncol=75))
csv<-data.frame(matrix(NA,nrow=rows,ncol=75))
colnames(csv) <- c("SeqIn","NumTiny","NumSmall","NumAliphatic","NumAromatic","NumNonPolar","NumPolar","NumCharged","NumBasic","NumAcidic","PorcTiny","PorcSmall","PorcAliphatic","PorcAromatic","PorcNonPolar","PorcPolar","PorcCharged","PorcBasic","PorcAcidic","at_index","at_boman","at_charge","at_pi","at_lengthpep","at_mw","at_hmoment_alpha","at_hmoment_sheet","HelixBendPreference","SideChainSize","ExtendedStructurePreference","Hidrophobicity","DoubleBendPreference","PartialSpecificVolume","FlatExtendedPreference","OccurrenceInAlphaRegion","pKC","SurroundingHidrophobicity","Blosum1","Blosum2","Blosum3","Blosum4","Blosum5","Blosum6","Blosum7","Blosum8","Blosum9","Blosum10","MsWhim1","MsWhim2","MsWhim3","st1","st2","st3","st4","st5","st6","st7","st8","t1","t2","t3","t4","t5","z1","z2","z3","z4","z5","HydrophobicityIndex","AlphaAndTurnPropensities","BulkyProperties","CompositionalCharacteristicIndex","LocalFlexibility","ElectronicProperties","Class")
#csv$SeqIn <- seqs #uno a uno las funciones


for (x in 1:rows) {
  print(x)
  y <- data.frame(aaComp(seqs[x]))
  kF <- data.frame(kideraFactors(seqs[x]))
  colnames(kF)<-"KF"
  bI <- data.frame(blosumIndices(seqs[x]))
  colnames(bI)<-"bI"
  mwS<-data.frame(mswhimScores(seqs[x]))
  colnames(mwS)<-"mwS"
  pFP <- data.frame(protFP(seqs[x]))
  colnames(pFP)<-"pFP"
  tS <- data.frame(tScales(seqs[x]))
  colnames(tS)<-"tS"
  zS <- data.frame(zScales(seqs[x]))
  colnames(zS)<-"zS"
  fV <- data.frame(fasgaiVectors(seqs[x]))
  colnames(fV)<-"fV"
  fila <- c(seqs[x],y$Number,y$Mole.,aIndex(seqs[x]),boman(seqs[x]),charge(seqs[x], pH = 7, pKscale = "Lehninger"),pI(seqs[x]),lengthpep(seqs[x]), mw(seqs[x]),hmoment(seqs[x], angle = 100, window = 11),hmoment(seqs[x], angle = 160, window = 11),kF$KF,bI$bI,mwS$mwS,pFP$pFP,tS$tS,zS$zS,fV$fV,"1")
  csv[x,] <- fila
}

#Hacer csv de descriptores

#write.table(csv, file = "descriptors_influenza.csv", append = FALSE, quote = TRUE, sep = ",", row.names = FALSE, col.names = TRUE)
#AAQVLSEMVMCGGS
#incluir NO EPITODPOS
#csv_NE = read.csv('Hemagglutinin.txt',sep = " ", col.names = "NE")
#csv_NE = read.csv('non-epi-influenza-non-red.txt',sep = " ", col.names = "NE",header = F)
csv_NE = read.csv('influenza_non-epitopes.txt',sep = " ", col.names = "NE",header = F)
seqs2 <- csv_NE$NE
seqs2[1]
rows <- nrow(csv_NE)
csvNE<-data.frame(matrix(NA,nrow=rows,ncol=75))
colnames(csvNE) <- c("SeqIn","NumTiny","NumSmall","NumAliphatic","NumAromatic","NumNonPolar","NumPolar","NumCharged","NumBasic","NumAcidic","PorcTiny","PorcSmall","PorcAliphatic","PorcAromatic","PorcNonPolar","PorcPolar","PorcCharged","PorcBasic","PorcAcidic","at_index","at_boman","at_charge","at_pi","at_lengthpep","at_mw","at_hmoment_alpha","at_hmoment_sheet","HelixBendPreference","SideChainSize","ExtendedStructurePreference","Hidrophobicity","DoubleBendPreference","PartialSpecificVolume","FlatExtendedPreference","OccurrenceInAlphaRegion","pKC","SurroundingHidrophobicity","Blosum1","Blosum2","Blosum3","Blosum4","Blosum5","Blosum6","Blosum7","Blosum8","Blosum9","Blosum10","MsWhim1","MsWhim2","MsWhim3","st1","st2","st3","st4","st5","st6","st7","st8","t1","t2","t3","t4","t5","z1","z2","z3","z4","z5","HydrophobicityIndex","AlphaAndTurnPropensities","BulkyProperties","CompositionalCharacteristicIndex","LocalFlexibility","ElectronicProperties","Class")
#csv$SeqIn <- seqs #uno a uno las funciones


for (x in 1:rows) {
  print(x)
  y <- data.frame(aaComp(seqs2[x]))
  kF <- data.frame(kideraFactors(seqs2[x]))
  colnames(kF)<-"KF"
  bI <- data.frame(blosumIndices(seqs2[x]))
  colnames(bI)<-"bI"
  mwS<-data.frame(mswhimScores(seqs2[x]))
  colnames(mwS)<-"mwS"
  pFP <- data.frame(protFP(seqs2[x]))
  colnames(pFP)<-"pFP"
  tS <- data.frame(tScales(seqs2[x]))
  colnames(tS)<-"tS"
  zS <- data.frame(zScales(seqs2[x]))
  colnames(zS)<-"zS"
  fV <- data.frame(fasgaiVectors(seqs2[x]))
  colnames(fV)<-"fV"
  fila <- c(seqs2[x],y$Number,y$Mole.,aIndex(seqs2[x]),boman(seqs2[x]),charge(seqs2[x], pH = 7, pKscale = "Lehninger"),pI(seqs2[x]),lengthpep(seqs2[x]), mw(seqs2[x]),hmoment(seqs2[x], angle = 100, window = 11),hmoment(seqs2[x], angle = 160, window = 11),kF$KF,bI$bI,mwS$mwS,pFP$pFP,tS$tS,zS$zS,fV$fV,"0")
  csvNE[x,] <- fila
}
#Agregar atributo clase 0 1
#por cada epitope una columna indica si es 0 o no 1 epitopo
#0 para epítopo y 1 para no epítopo.

#Hacer csv de todos los epitopes
#juntar dataframes
df3 = rbind(csv,csvNE)
write.table(df3, file = "descriptors_class_influenza_nonEpiIEDB.csv", append = FALSE, quote = TRUE, sep = ",", row.names = FALSE, col.names = TRUE)

#correr

