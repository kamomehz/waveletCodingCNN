/*
 * predictKa.c, inspired from 
 * 
 *  predict.c
 *  Created on: 2014/07/24
 *      Author: yamasita
 *      For block matching and linear prediction
 */



/* Normalize Low coef by dividing 2^0.5 */
#define STAGERATIO 1.41421356


double predictNN();
double predictLine();

Classmaker *Init(){
    /* Load CNN parameters to prepared */
    int total = 1545281;
    prepared =  (double *) malloc(sizeof(double) * total);
    fileL = fopen(CNNData, "r");
    for(int aRead=0;aRead<total;aRead++) fscanf(fileL,"%lf,",&prepared[aRead]);
    fclose(fileL);

    /* malloc hidden layers and zeroPadding matrix */
    int size=blockWidth*blockHeight;
    c1 =  (double *) malloc(sizeof(double) * (32*size));
    c2 =  (double *) malloc(sizeof(double) * (128*size));
    c3 =  (double *) malloc(sizeof(double) * (512*size));
    f2 =  (double *) malloc(sizeof(double) * (64));
    kernel =  (double *) malloc(sizeof(double) * (3*3));
    zeroPad =  (double *) malloc(sizeof(double) * (blockWidth+2)*(blockHeight+2));

	return pred;
}


predictLine{
	double *block;
	double norm;

    /*  LH HL use CNN */
    if(LH || HL){ 
        /*  skip top 3 lines */
        if (y < 4) {
            out[all] = 0.0;
            return; 
        }
        // 0 paddings on left(1) and right(2):
        left padding: out[-1,-2] = 0.0;
        right padding: out[1] = 0.0;
        
        for (creatBlock) {
            for (res) {
                pos       = position(Restored);
                block[pos] = Res[pos]/norm;
            }
            for (low) {
                pos       = position(Low);
                block[pos] = Low[pos]/stageRatio/norm;
            }
            out[ixH++] = predictNN()*norm;
        }
    }
    out[ix] = 0.0;
}



predictNN(block){

	//parameter scanner
	int scan=0; 
	int size=nBlockData;

	// initialize all layer storage
    for (int i=0; i<32*size;i++) c1[i]=0;
    for (int i=0; i<128*size;i++) c2[i]=0;
    for (int i=0; i<512*size;i++) c3[i]=0;
    for (int i=0; i<64;i++) f2[i]=0;

	double output=0;
	double *zeroPad=zeroPad;

    // c1 conv // int j -> input dimension ~ 1 at c1
    for (int i=0; i<32;i++){
        // kernel=3
        for (int ki=0; ki<3*3; ki++)kernel[ki]=prepared[scan++];
        // zeroPad ready to conv
        padding(block, zeroPad, blockWidth, blockHeight);
        // conv: (9*6 + 3*3 ~ size:=28)
        conv2d(zeroPad, kernel, &(c1[i*size]), blockWidth, blockHeight);
    }

    // c1 bias
    for (int i=0; i<32;i++){
        for(int j=0; j<size;j++){
            c1[i*size+j]+=prepared[scan];
            //activate
            c1[i*size+j]=(c1[i*size+j] >= 0) ? c1[i*size+j] : 0.01*c1[i*size+j] ;
        }
        scan++;
    }

    // c2 conv
    for (int i=0; i<128;i++){
        for (int j=0; j<32;j++){
            padding(&(c1[j*size]), zeroPad, blockWidth, blockHeight);
            for (int ki=0; ki<3*3; ki++) kernel[ki]=prepared[scan++];
            conv2d(zeroPad, kernel, &(c2[i*size]), blockWidth, blockHeight);
        }
    }

    // c2 bias
    for (int i=0; i<128;i++){
        for(int j=0; j<size;j++){
            c2[i*size+j]+=prepared[scan];
            c2[i*size+j]=(c2[i*size+j] >= 0) ? c2[i*size+j] : 0.01*c2[i*size+j] ;
        }
        scan++;
    }

    // c3 conv
    for (int i=0; i<512;i++){
        for (int j=0; j<128;j++){
            padding(&(c2[j*size]), zeroPad, blockWidth, blockHeight);
            for (int ki=0; ki<3*3; ki++) kernel[ki]=prepared[scan++];
            conv2d(zeroPad, kernel, &(c3[i*size]), blockWidth, blockHeight);
        }
    }

    // c3 bias
    for (int i=0; i<512;i++){
        for(int j=0; j<size;j++){
            c3[i*size+j]+=prepared[scan];
            c3[i*size+j]=(c3[i*size+j] >= 0) ? c3[i*size+j] : 0.01*c3[i*size+j] ;
        }
        scan++;
    }


    // c3==f1 to fc2
    for (int i=0; i<64; i++){
        for (int j=0; j<size*512; j++){
            f2[i] += c3[j] * prepared[scan++];
        }
    }

    for (int i=0; i<64;i++){
        f2[i]+=prepared[scan];
        f2[i]=(f2[i] >= 0) ? f2[i] : 0.01*f2[i] ;
        scan++;
    }
    

    // last
    for (int j=0; j<64; j++){
        output += f2[j] * prepared[scan++];
    }
    output += prepared[scan];

	return output;
}


// in: [w,h]; out: [w+1,h+1]
void padding{
  int i,j;
  // updown 0
  for (j=0;j<w+2;j++) {
    out[j]=0;
    out[(h+1)*(w+2)+j]=0;
  }
  // Left Right 0
  for (i=0;i<h+2;i++) {
    out[i*(w+2)]=0;
    out[i*(w+2)+w+1]=0;
  }
  for (i=0;i<h;i++){
    for (j=0;j<w;j++){
      out[(i+1)*(w+2)+j+1]=in[i*w+j];
    }
  }
}

// in: [w+1,h+1]; out: [w,h]
conv2d(int w, int h){
	int i,j;
	int n,m;
	int k=3;
    double sum;
	for(i=0; i<h; i++)
	for(j=0; j<w; j++){
        sum = 0.0;
        for(m=0; m<k; m++)
        for(n=0; n<k; n++){
            sum += kernel[n*k+m] * in[(i+n)*(w+2)+j+m];
        }
        out[i*w+j]+=sum;
    }
}