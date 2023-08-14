%%cu

#include<bits/stdc++.h>
#include<cuda.h>
#include <sys/time.h>
#define bias -0.3
#define n 1024
#define IM 60000
#define lim 60000
#define mx_layer 120

#define layer_loc "/content/drive/MyDrive/data/neuron1024/"
#define category_loc "/content/drive/MyDrive/data/"
#define feature_loc "/content/drive/MyDrive/data/"

using namespace std;

__global__ void CSR_SpMV(int *cm,int *row,float *val,float *d_Y,int* all_zero) {
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    
    __shared__ float Y[n];
    Y[ind]  = d_Y[ind];

    __syncthreads();

    float sum=0.0;
    for(int i=cm[ind];i<cm[ind+1];i++){
        //printf("%d\n",ptr);
        sum+=Y[row[i]]*val[i]*1.0;
    }
    __syncthreads();
    
    if(Y[ind]>=0.0)
      Y[ind]=sum+bias;

    //ReLU operation
    if(Y[ind]<0.0)Y[ind]=0.0;
    else if(Y[ind]>32.0)Y[ind]=32.0;

    if(Y[ind] != 0 && *all_zero == 1){
        *all_zero = 0;
    }
    d_Y[ind]  = Y[ind];
    
}

class Layercls {
    public:
        int *cm;
        int *row;
        int nnz;
        float *val;
        void add_layer(vector<Layercls> &lvec,Layercls &ly){
            lvec.push_back(ly);
        }
};

void store_layers(int max_layer, vector<Layercls> &layers ){
    
    for(int lay_no=1;lay_no<=max_layer;lay_no++){
      float* M;
      M = (float*)calloc(n*n,sizeof(float));
      int *cm, *row;
      float *val;
      string fname(layer_loc);
      vector<string> fpara={"n",to_string(n), "-l", to_string(lay_no), ".tsv"};
      for(string x:fpara){
          fname.append(x);
      }
      ifstream file(fname);

      string line;
      int NNZ=0;
      while(getline(file,line)){
          stringstream ss(line);
          int r,c;
          float v;
          ss >> r >> c >> v;
          r--;c--;
          M[r*n+c]=v;
          NNZ++;
      }
      cm=(int*)malloc((n+1)*sizeof(int));
      row=(int*)malloc(NNZ*sizeof(int));
      val=(float*)malloc(NNZ*sizeof(float));

      cm[0]=0;
      int ptr=0,nnz_cnt=0;
      for(int j=0;j<n;j++){
          nnz_cnt=0;
          for(int i=0;i<n;i++){
              if(M[i*n+j]!=0.0){
                  //cout<<i<<" "<<j<<" "<<M[i*n+j]<<" "<<ptr<<endl;
                  val[ptr]=M[i*n+j];
                  row[ptr]=i;
                  nnz_cnt++;
                  ptr++;
              }
          }
          cm[j+1]=cm[j]+nnz_cnt;
      }
      Layercls l1;
      l1.cm=cm;
      l1.row=row;
      l1.val=val;
      l1.nnz=NNZ;
      l1.add_layer(layers,l1);
      delete M;
    }
    
}
void load_feature_vector(vector<float*> &all_feature_vec){
    string fname(feature_loc);
    fname.append("sparse-images-1024.tsv");
    ifstream file(fname);
    float *all;
    all=(float*)calloc(IM*n,sizeof(float));
    string line;
    while(getline(file,line)){
        stringstream ss(line);
        int r,c;
        float val;
        ss >> r >> c >> val;
        r--;c--;
        all[r*n+c]=val;
    }

    for(int i=0;i<IM;i++){
        float *img;
        img=(float*)calloc(n,sizeof(float));
        for(int j=0;j<n;j++){
            img[j]=all[i*n+j];
        }
        all_feature_vec.push_back(img);
    }
    delete all;
}

void load_true_category(vector<int> &true_cat){
    string fname(category_loc);
    fname.append("neuron1024-l"+to_string(mx_layer)+"-categories.tsv");
    ifstream file(fname);
    string line;
    while(getline(file,line)){
        stringstream ss(line);
        int cat;
        ss >> cat;
        if(cat>lim) break;
        true_cat.push_back(cat);
    }
}

int main(){
    
    vector<Layercls> layers;
    store_layers(mx_layer,layers);

    vector<float*> all_img;
    load_feature_vector(all_img);
    

    int *d_cm, *d_row;
    
    cudaMalloc(&d_cm,(n+1)*sizeof(int));
    cudaMalloc(&d_row,n*n*sizeof(int));

    float *d_val;
    cudaMalloc(&d_val,n*n*sizeof(float));
    

    float *Y;
    Y=(float*)malloc(n*sizeof(float));

    int *all_zero;
    int *d_all_zero;
    cudaMalloc(&d_all_zero,sizeof(int));
    
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    /* Timer Starts */

    vector<bool> is_cat(lim,true);

    vector<float*> d_img;
    for(int img=0;img<lim;img++){
        float* d_Y;
        cudaMalloc(&d_Y,n*sizeof(float));
        cudaMemcpy(d_Y,all_img[img],n*sizeof(float),cudaMemcpyHostToDevice);
        d_img.push_back(d_Y);
    }

    for(int lay=0;lay<mx_layer;lay++){
        int NNZ=layers[lay].nnz;
        cudaMemcpy(d_cm,layers[lay].cm,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_row,layers[lay].row,NNZ*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_val,layers[lay].val,NNZ*sizeof(float),cudaMemcpyHostToDevice);
        
        for(int img=0;img<lim;img++){
          if(!is_cat[img])continue;
          *all_zero = 1;
          cudaMemcpy(d_all_zero,all_zero,sizeof(int),cudaMemcpyHostToDevice);
          CSR_SpMV<<<1,n>>> (d_cm,d_row,d_val,d_img[img],d_all_zero);
          cudaMemcpy(all_zero,d_all_zero,sizeof(int),cudaMemcpyDeviceToHost);
          if(*all_zero == 1) is_cat[img]=false;
        }
    }
    vector<int> cat;
    for(int i=0;i<lim;i++){
        if(is_cat[i]) cat.push_back(i+1);
    }
    /* Timer Stops */
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    
    vector<int> true_cat;
    load_true_category(true_cat);

    // category comparison
    bool pas=(cat.size()==true_cat.size());
    for(int i=0;i<cat.size();i++){
        if(cat[i]!=true_cat[i]){
            pas=false;
            break;
        }
    }
    if(pas){
        float ssize=(lim*1.0/IM)*100;
        printf("Test Passed! \nSample size used: %f % \nTime measured: %.6f seconds.\n",ssize, elapsed);
    }
    else{
        printf("Test failed!!");
    }
    
    return 0;
}


