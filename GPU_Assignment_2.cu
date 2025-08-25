#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
//#include <cuda/cuda_runtime.h>     //Commented it out as it is not working in my device.

#define SIZE 32 //define the tile size

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

//declare the __constant__ global memory(read-only) for filter
__constant__ long int d_filter[4096];

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    // sample kernel you can use your own kernel
    //Shared Matrix of size (SIZE+r-1)*(SIZE+s-1) with the required padding(r/2 in length & s/2 in width on both sides) for SIZE*SIZE tile
    extern __shared__ long int shmatrix[];

    //Indices of the focus element in Global Matrix
    int mat_actual_row=blockIdx.y*SIZE+threadIdx.x;
    int mat_actual_col=blockIdx.z*SIZE+threadIdx.y;

    //Indices of the focus element in Shared Matrix(with padding)
    int sh_actual_row=r/2+threadIdx.x;
    int sh_actual_col=s/2+threadIdx.y;

    int sh_it_row,sh_it_col,sh_it_ind,mat_it_row,mat_it_col,mat_it_ind;
    long long sum=0;
    //Iterate over the channels
    for(int chno=0;chno<c;chno++)
    {
        //calculate 1-dimensional index for shared iterative element
        sh_it_ind=threadIdx.x*SIZE+threadIdx.y;
        //Loading the matrix from global memory into shared memory
        while(sh_it_ind<(SIZE+r-1)*(SIZE+s-1))
        {
            //calculate 2-dimensional index for shared iterative element
            sh_it_row=sh_it_ind/(SIZE+s-1);
            sh_it_col=sh_it_ind%(SIZE+s-1);

            //calculate the corresponding indices for element in shared matrix in actual matrix 
            mat_it_row=mat_actual_row-(sh_actual_row-sh_it_row);
            mat_it_col=mat_actual_col-(sh_actual_col-sh_it_col);

            //checking for the out of bound in actual matrix
            if(mat_it_row<0 || mat_it_row>=h || mat_it_col<0 || mat_it_col>=w)
            {
                shmatrix[sh_it_ind]=0;
            }
            else
            {
                mat_it_ind=mat_it_row*w+mat_it_col;
                shmatrix[sh_it_ind]=matrix[chno*h*w+mat_it_ind];
            }
            sh_it_ind+=SIZE*SIZE;
        }
        //ensure that all threads load memory into shared memory
        __syncthreads();
        //compute the convolution
        for(int i=0;i<r;i++)
        {
            sh_it_row=sh_actual_row-r/2+i;
            for(int j=0;j<s;j++)
            {
                sh_it_col=sh_actual_col-s/2+j;
                sum+=shmatrix[sh_it_row*(SIZE+s-1)+sh_it_col]*d_filter[blockIdx.x*c*r*s+chno*r*s+i*s+j];
            }
        }
        //ensure that all threads conclude for a particular channel
        __syncthreads();
    }
    //storing the sum in corresponding element in result & block the unaligned boundary threads
    if(mat_actual_row<h && mat_actual_col<w)result[blockIdx.x*h*w+mat_actual_row*w+mat_actual_col]=sum;
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/
    //Allocate the device memory
    long int *d_mat;
    long int *d_ans;
    cudaMalloc(&d_mat,c*h*w*sizeof(long int));
    cudaMalloc(&d_ans,k*h*w*sizeof(long int));

    //copy to the device memory
    cudaMemcpy(d_mat,h_mat,h*w*c*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter,h_filter,r*s*c*k*sizeof(long int),0,cudaMemcpyHostToDevice);

    //specify the grid,block dimensions
    dim3 blocksPerGrid(k,(h+SIZE-1)/SIZE,(w+SIZE-1)/SIZE);
    dim3 threadsPerBlock(SIZE,SIZE,1);
    //launch the kernel
    dkernel<<<blocksPerGrid, threadsPerBlock,(SIZE+r-1)*(SIZE+s-1)*sizeof(long int)>>>(d_mat,d_filter,d_ans,h,w,c,r,s,k);

    //copying back the result matrix into host memory
    cudaMemcpy(h_ans,d_ans,k*h*w*sizeof(long int),cudaMemcpyDeviceToHost);
    //free the device memory
    cudaFree(d_mat);
    cudaFree(d_ans);

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
