# plant_diseases

*PIPELINE*:  
LOAD IMAGE 256 x 256  
NORMALIZE TO NONZERO STATS: *MEAN RED = 85.384*; WITH *STD = 53.798*  

**Extracting features on Tomato:**  
*Time:* 45 min  
*Train feat size:* 10.7 G  
*Test feat size:* 3.6 G  
 
-> MASK 17 x 17 -> LOCAL 4(8) LEVEL QUANTIZATION USING MIN,STD,MEAN,MAX (d/2) VALUES
STRIDE 2(4) -> 112 x 112 (56 x 56) x N_FEAT.
