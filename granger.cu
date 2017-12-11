#include <fstream>
#include <string>
#include <iostream>
#include <dirent.h>
#include <sstream>
#include <cstring>
#include <math.h>
#include <vector>
#include <algorithm>

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <vector>

#include <cuda_runtime.h>
//#include "cusparse_v2.h"
//#include "cusparse.h"
#include <iomanip>

using namespace std;

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

// c++ code to finding the eigenvalue using power iteration
float power_iter(float *A, float *start, int power, int iter)
{
	float *c = new float[power];
	float d = 0.0f;
	float temp = 0.0f;
	int i,j;
	for (int ii = 0; ii < 30; ii++)
	{
		for(i= 0 ; i < power; i++)
		{
			c[i]=0.0f;
			for(j = 0; j < power; j++)
				c[i] += A[i*power + j]*start[j];
		}
		for(i = 0;i < power; i++)
			start[i] = c[i];

		temp = d;
		d = 0.0f;

		for(i = 0; i < power; i++)
		{
			if(fabs(start[i])>fabs(d))
				d = start[i];
		}
		for(i = 0; i < power; i++)
			start[i]/=d;
		if(fabs(d-temp) < 0.00001 && iter > 20)
		{
			delete[] c;
			return d;
		}
	}
	delete[] c;
	return d;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// error checking in gpu
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// CUDA kernel for matrix multiplication

// data reading by nicolas
float** createF(int a, int b){
	float** r=new float*[a];
	for(int i=0;i<a;i++){
		r[i]=new float[b];
		for(int j=0;j<b;j++){
			r[i][j]=0.0f;
		}
	}
	return r;
}

// data reading by nicolas
float** loadData(string file, int* r){
	std::ifstream FileIn((char*)file.c_str(), std::ios_base::binary);
	// reading file size
	int a=0;
	int b=0;
	FileIn.read(reinterpret_cast<char*>(&a), sizeof(int));
	FileIn.read(reinterpret_cast<char*>(&b), sizeof(int));
	r[0]=a;
	r[1]=b;
	// reading file content
	float** signals=createF(a,b);
	float f=0.0f;
	for(int x=0;x<a;x++){
		for(int t=0;t<b;t++){
			FileIn.read(reinterpret_cast<char*>(&f), sizeof(float));
			signals[x][t]=f;
		}
	}
	FileIn.close();
	return signals;
}

// saving data by nicolas
void saveData(float** signals, string file, int a,int b){
	std::ofstream FileOut((char*)file.c_str(), std::ios_base::binary);
	FileOut.write(reinterpret_cast<char*>(&a), sizeof(int));
	FileOut.write(reinterpret_cast<char*>(&b), sizeof(int));
	for(int x=0;x<a;x++){
		for(int t=0;t<b;t++){
			FileOut.write(reinterpret_cast<char*>(&signals[x][t]), sizeof(float));
		}
	}
	FileOut.close();
}

// get data in vector format
float* get_data1(const int nodes, const int subjects, const int time_points)
{

	// initialize the data
	float* data = new float[nodes*subjects*time_points*4];
	for (int i = 0; i < nodes*subjects*time_points*4; i++)
	{
		data[i] = 0.0f;
	}

	DIR *dir;

	struct dirent *ent;

	// open directory
	if ((dir = opendir("/home/sharath/dushyant/180/fonctional/")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {

			string str = "/home/sharath/dushyant/180/fonctional/";

			// append the file name
			string file = str.append(ent->d_name);


			if (strlen(ent->d_name) > 3)
			{

				istringstream ss(ent->d_name);
				string token;
				int temp = 0;
				int* where = new int[4];
				for(int i=0;i<4;i++){
					where[i]=0;
				}

				// break the name around '.'
				while(std::getline(ss, token, '.')) {
					if (temp == 2)
					{
						if(token == "R")
						{
							where[temp] = 1;
						}
						else
						{
							where[temp] = 0;
						}
					}
					else
					{
						stringstream(token) >> where[temp];
					}


					temp = temp + 1;
				}

				int* r=new int[2];
				for(int i=0;i<2;i++){
					r[i]=0;
				}

				float** temp_data = loadData(file,r);
				if ((where[0]-1) < subjects){
					// add time series of only one scan (total 4 scans)
					//if (where[1] == 1)
					//{
					for (int i = 0; i < 180 ; i++)
					{
						for (int j = 0; j < time_points; j++)
						{
							data[((where[2]*180+i)*subjects*time_points*4) + ((where[0]-1)*time_points*4) + j+(where[1]-1)*time_points] = temp_data[i][j];
						}
					}

				}
				delete [] r;
				for (int i = 0; i < 180; i++)
				{
					delete[] temp_data[i];
				}
				delete [] temp_data;
				//cout << "here is the number" << where[1] << "\n";

				//}

			}

		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");

	}
	//cout << data[0*subjects*time_points*4 + 0*time_points*4 + 2] << "\n";
	//cout << data[0*subjects*time_points*4 + 0*time_points*4 + 1] << "\n";
	//cout << data[0*subjects*time_points*4 + 0*time_points*4 + 0] << "\n";
	//cout << data[2*subjects*time_points*4 + 0*time_points*4 + 0] << "\n";
	//cout << data[0*subjects*time_points*4 + 1*time_points*4 + 0] << "\n";


	return data;
}

// get data in vector format for half of the time series for training
float* get_data2(const int nodes, const int subjects, const int time_points)
{

	// initialize the data
	float* data = new float[nodes*subjects*time_points*2];
	for (int i = 0; i < nodes*subjects*time_points*2; i++)
	{
		data[i] = 0.0f;
	}

	DIR *dir;

	struct dirent *ent;

	// open directory
	if ((dir = opendir("/home/sharath/dushyant/180/fonctional/")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {

			string str = "/home/sharath/dushyant/180/fonctional/";

			// append the file name
			string file = str.append(ent->d_name);


			if (strlen(ent->d_name) > 3)
			{

				istringstream ss(ent->d_name);
				string token;
				int temp = 0;
				int* where = new int[4];
				for(int i=0;i<4;i++){
					where[i]=0;
				}

				// break the name around '.'
				while(std::getline(ss, token, '.')) {
					if (temp == 2)
					{
						if(token == "R")
						{
							where[temp] = 1;
						}
						else
						{
							where[temp] = 0;
						}
					}
					else
					{
						stringstream(token) >> where[temp];
					}


					temp = temp + 1;
				}

				int* r=new int[2];
				for(int i=0;i<2;i++){
					r[i]=0;
				}

				float** temp_data = loadData(file,r);
				if ((where[1]-1) >= 2){
					// add time series of only one scan (total 4 scans)
					//if (where[1] == 1)

					//{
					for (int i = 0; i < 180 ; i++)
					{
						for (int j = 0; j < time_points; j++)
						{
							if ((where[0]) < (subjects+1))
							{
								data[((where[2]*180+i)*subjects*time_points*2) + ((where[0]-1)*time_points*2) + j+(where[1]-3)*time_points] = temp_data[i][j];
							}
						}
					}

				}
				delete [] r;
				for (int i = 0; i < 180; i++)
				{
					delete[] temp_data[i];
				}
				delete [] temp_data;
				//cout << "here is the number" << where[1] << "\n";

				//}

			}

		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");

	}


	return data;
}

// get data in vector format from other half of time series for testing
float* get_data3(const int nodes, const int subjects, const int time_points)
{

	// initialize the data
	float* data = new float[nodes*subjects*time_points*2];
	for (int i = 0; i < nodes*subjects*time_points*2; i++)
	{
		data[i] = 0.0f;
	}

	DIR *dir;

	struct dirent *ent;

	// open directory
	if ((dir = opendir("/home/sharath/dushyant/180/fonctional/")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {

			string str = "/home/sharath/dushyant/180/fonctional/";

			// append the file name
			string file = str.append(ent->d_name);


			if (strlen(ent->d_name) > 3)
			{

				istringstream ss(ent->d_name);
				string token;
				int temp = 0;
				int* where = new int[4];
				for(int i=0;i<4;i++){
					where[i]=0;
				}

				// break the name around '.'
				while(std::getline(ss, token, '.')) {
					if (temp == 2)
					{
						if(token == "R")
						{
							where[temp] = 1;
						}
						else
						{
							where[temp] = 0;
						}
					}
					else
					{
						stringstream(token) >> where[temp];
					}


					temp = temp + 1;
				}

				int* r=new int[2];
				for(int i=0;i<2;i++){
					r[i]=0;
				}

				float** temp_data = loadData(file,r);
				if ((where[1]-1) < 2){
					// add time series of only one scan (total 4 scans)
					//if (where[1] == 1)

					//{
					for (int i = 0; i < 180 ; i++)
					{
						for (int j = 0; j < time_points; j++)
						{
							if ((where[0]) < (subjects+1))
							{
								data[((where[2]*180+i)*subjects*time_points*2) + ((where[0]-1)*time_points*2) + j+(where[1]-1)*time_points] = temp_data[i][j];
							}
						}
					}

				}
				delete [] r;
				for (int i = 0; i < 180; i++)
				{
					delete[] temp_data[i];
				}
				delete [] temp_data;
				//cout << "here is the number" << where[1] << "\n";

				//}

			}

		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");

	}

	return data;
}

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

// check if file exist
bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

// generate a sparse random matrix represented in vector format
float* sparse_generate_matrix1( int a,  int b, float sparse)
{
	float *random = new float[a*b];
	for (int i = 0; i < a*b; i++)
	{
		random[i] = 0.0f;
	}

	for(int o = 0; o < a; o++)
	{
		float *temp = new float[b];
		for(int k = 0; k < b; k++)
		{

			temp[k] = 0.0f;
		}

		for(int i = 0; i < floor(b*sparse); i++)
		{

			temp[i] = ((float) rand()) / (float) RAND_MAX;
		}
		random_shuffle(temp, temp + b);
		for (int j = 0; j < b; j++)
		{
			random[o*b+j] = temp[j];

		}
		delete[] temp;

	}
	return random;
}

// generate sparse and diagonal matrix B
float* random_B( int d, int n_tau)
{

	float *B = new float[d*d*n_tau];
	
	for ( int tau = 0; tau < n_tau; tau++)
	{
		float* temp_b = sparse_generate_matrix1( d , d , 0.2);
		for (int p = 0; p < d; p++)
		{
			for (int q = 0; q < d; q++)
			{
				B[p*d*n_tau + q*n_tau + tau] = 0.3f*temp_b[p*d + q];
				if (p==q)
				{
					B[p*d*n_tau + q*n_tau + tau] = B[p*d*n_tau + q*n_tau + tau] + 0.4f;
				}
			}
		}
		delete[] temp_b;
	} 
/*
	for ( int tau = 0; tau < n_tau; tau++)
	{
		for (int p = 0; p < d; p++)
		{
			for (int q = 0; q < d; q++)
			{
				cout <<B[p*d*n_tau + q*n_tau + tau] << ",";
			}
		}
		cout << "\n";
	}
*/
	return B;
}

// generate subject specific matrices 
float* random_a(int n_i, int n_tau)
{
	float *a = new float[n_i*n_tau];
	for (int i = 0; i < n_i; i++)
	{
		for (int tau = 0; tau < n_tau; tau++)
		{

			a[i*n_tau + tau] = RandomFloat(0.5f,0.6f)/(float(tau)+1.0f);
		}
	}
	return a;
}


// generate A matrix
float* generate_A(float *B, int n_i, int d, int n_tau)
{

	float *a = random_a(n_i, n_tau);

	float *A = new float[d*d*n_i*n_tau];
	for (int i = 0; i < n_i; i++)
	{
		for (int tau = 0; tau < n_tau; tau++)
		{

			for (int p = 0; p < d; p++)
			{
				for (int q = 0; q < d; q++)
				{
					A[p*d*n_i*n_tau + q*n_i*n_tau + i*n_tau + tau] = a[i*n_tau + tau]*B[p*d*n_tau + q*n_tau + tau];		
				}
			}
		}
	}

	delete[] a;

	return A;
}

double rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

float* generate_X(float *B, int d, int n_i, int n_tau, int n_t)
{
	int temp_t = 20;
	float *A = generate_A(B, n_i, d,n_tau);

	float *error = new float[(n_t+temp_t)];
	for (int t = 0; t < (n_t+temp_t); t++)
	{
		srand(time(NULL));
		error[t] = rand_normal(0.0f, 1.0f);	
	}

	float *temp_X = new float[d*n_i*(n_t+temp_t)];
	float *X = new float[d*n_i*n_t];
	for (int i = 0; i < n_i; i++)
	{
		for (int t = 0 ; t < (n_t+temp_t) ; t++)
		{
			for (int p = 0; p < d; p++)
			{
				temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t] = 0.0f;
				for (int tau = 0; tau < n_tau; tau++)
				{
					for (int q = 0; q < d; q++)
					{
						if (t >= n_tau)
						{
							temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t] = temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t] + A[p*d*n_i*n_tau + q*n_i*n_tau + i*n_tau + tau]*temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t-n_tau];
						}
					}
				}
				temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t] = temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t] + error[t];
			}
		}
	}

	for (int i = 0; i < n_i; i++)
	{
		for (int t = temp_t ; t < (n_t+temp_t) ; t++)
		{
			for (int p = 0; p < d; p++)
			{
				X[p*n_i*n_t + i*n_t + t-temp_t] = temp_X[p*n_i*(n_t+temp_t) + i*(n_t+temp_t) + t];
			}
		}
	}

	delete[] error;
	delete[] A;
	delete[] temp_X;
	return X;
}

// generate a random vector of size a*b*c
float* generate_matrix3_1(int a, int b, int c)
{
	float *random = new float[a*b*c];

	for(int o = 0; o < a; o++)
	{
		for(int i = 0; i < b; i++)
		{
			for (int p = 0; p < c; p++)
			{
				random[o*b*c + i*c + p] = ((float) rand()) / (10.f*(float) RAND_MAX);
			}
		}
	}
	return random;
}


// calculate the ||x - x_hat||2^2
void calc_norm(float* data, float* a, float*b, float* c, int n_i, int n_k, int n_t, int d, int n_tau, float *temp_c)
{

	float temp_b = 0.0f;
	temp_c[0] = 0.0f;
	for (int i = 0; i < n_i; i++)
	{
		for (int t = n_tau; t < n_t; t++)
		{
			for (int p = 0; p < d; p++)
			{
				temp_b = 0.0f;
				for (int tau = 0; tau < n_tau; tau++)
				{
					for (int k = 0; k < n_k; k++)
					{
						for (int q = 0; q < d; q++)
						{
							temp_b = temp_b + a[i*n_k*n_tau + k*n_tau + tau]*b[k*d + p]*c[k*d + q]*data[q*n_i*n_t + i*n_t + t-tau-1];
						}
					}
				}
				temp_c[0] = temp_c[0] + ((data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t] - temp_b));
			}
		}

	}

}

// calculate the ||x - x_hat||2^2
__global__ void gpu_calc_norm(float* data, float* a, float*b, float* c, int n_i, int n_k, int n_t, int d, int n_tau, float *temp_c)
{
	temp_c[0] = 0.0f;
	float temp_b = 0.0f;
	unsigned i = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned p = threadIdx.y+blockDim.y*blockIdx.y;

	if (i >= n_i) return;
	if (p >= d) return;
	for (int t = n_tau; t < n_t; t++)
	{

		float x;

		temp_b = 0.0f;
		for (int tau = 0; tau < n_tau; tau++)
		{

			for (int k = 0; k < n_k; k++)
			{
				for (int q = 0; q < d; q++)
				{
					temp_b = temp_b + a[i*n_k*n_tau + k*n_tau + tau]*b[k*d + p]*c[k*d + q]*data[q*n_i*n_t + i*n_t + t-tau-1];
				}
			}
		}
		x = ((data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t ] - temp_b));
		atomicAdd(&temp_c[0], x);
	}

}

__global__ void gpu_calc_norm1(float* data, float* a, float*b, float* c, int n_i, int n_k, int n_t, int d, int n_tau, float *temp_c, int i, int t, int p)
{
	temp_c[0] = 0.0f;
	float temp_b;

	temp_b = 0.0f;
	for (int tau = 0; tau < n_tau; tau++)
	{

		for (int k = 0; k < n_k; k++)
		{
			for (int q = 0; q < d; q++)
			{
				temp_b = temp_b + a[i*n_k*n_tau + k*n_tau + tau]*b[k*d + p]*c[k*d + q]*data[q*n_i*n_t + i*n_t + t-tau-1 ];
			}
		}
	}
	temp_c[0] = temp_c[0] + ((data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t ] - temp_b));
}


void cpu_indvi_norms(float* data, float* a, float* b, float* sigma, int n_i, int n_k, int n_t, int d, int n_tau, float *g1)
{
	float temp_b;
	g1[0] = 0.0f;
	for (int i = 0; i < n_i; i++)
	{
		for (int t = n_tau; t < n_t; t++)
		{
			for (int p = 0; p < d; p++)
			{

				temp_b = 0.0f;
				for (int tau = 0; tau < n_tau; tau++)
				{
					for (int m = 0; m < n_k; m++)
					{
						temp_b = temp_b + b[m*d + p]*a[i*n_k*n_tau + m*n_tau + tau]*sigma[m*n_i*n_t + i*n_t + (t-tau-1)];
					}
				}


				g1[0] = g1[0] + (data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t] - temp_b);

			}
		}
	}

}
__global__ void indvi_norms(float* data, float* a, float* b, float* sigma, int n_i, int n_k, int n_t, int d, int n_tau, float *g1)
{
	float temp_b;
	g1[0] = 0.0f;
	float x;
	unsigned p = threadIdx.x+blockDim.x*blockIdx.x;

	if (p >= d) return;
	for (int t = n_tau; t < n_t; t++)
	{
		for (int i = 0; i < n_i; i++)
		{

			temp_b = 0.0f;
			for (int tau = 0; tau < n_tau; tau++)
			{
				for (int m = 0; m < n_k; m++)
				{
					temp_b = temp_b + b[m*d + p]*a[i*n_k*n_tau + m*n_tau + tau]*sigma[m*n_i*n_t + i*n_t + (t-tau-1)];
				}
			}
			x = (data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t] - temp_b);
			atomicAdd(&g1[0],x);

		}
	}
}

__global__ void indvi_norms1(float* data, float* a, float* b, int n_i, int n_t, int d, int n_tau, float *g1)
{
	float temp_b;
	g1[0] = 0.0f;
	float x;
	unsigned p = threadIdx.x+blockDim.x*blockIdx.x;

	if (p >= d) return;
	for (int t = n_tau; t < n_t; t++)
	{
		for (int i = 0; i < n_i; i++)
		{

			temp_b = 0.0f;
			for (int tau = 0; tau < n_tau; tau++)
			{
				for (int q = 0; q < d; q++)
				{
					temp_b = temp_b + a[i*n_tau + tau]*b[p*d*n_tau + q*n_tau+tau]*data[q*n_i*n_t + i*n_t + t-tau];
				}
			}
			x = (data[p*n_i*n_t + i*n_t + t] - temp_b)*(data[p*n_i*n_t + i*n_t + t] - temp_b);
			atomicAdd(&g1[0],x);

		}
	}
}

// generate H

__global__ void gpu_generate_H(float *a, float *data, float *H, int n_t, int n_i, int n_tau, int d)
{

	unsigned tau = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned q = threadIdx.y+blockDim.y*blockIdx.y;
	unsigned qb = threadIdx.z+blockDim.z*blockIdx.z;
	if (tau >= n_tau) return;
	if (q >= d) return;
	if (qb >= d) return;
			for (int taub = 0; taub < n_tau; taub++)
			{
			
					float temp = 0.0f;
					for (int i = 0; i < n_i; i++)
					{
						for (int t = n_tau; t < n_t; t++)
						{
							temp = temp + a[i*n_tau + tau]*data[q*n_i*n_t + i*n_t + t - tau]*a[i*n_tau + taub]*data[qb*n_i*n_t + i*n_t + t-taub]		;					
						}
					}
					H[tau*d*n_tau*d + q*n_tau*d + taub*d + qb] = temp;				
				}
	
}


// generate derivate of a
__global__ void gpu_dell_a(float *G, float *a, float *F, float *dell_a, int n_i, int n_tau)
{
	unsigned i = threadIdx.x+blockDim.x*blockIdx.x;
	if (i >= n_i) return;
		for (int tau = 0; tau < n_tau; tau ++)
		{
			float temp = 0.0f;
			for (int taub = 0; taub < n_tau; taub ++)
			{
				temp = temp + a[i*n_tau + taub]*F[i*n_tau*n_tau + taub*n_tau + tau];
			}
			dell_a[i*n_tau + tau] = 2.0f*(temp - G[i*n_tau + tau]);
		}
}


// generate G

__global__ void gpu_generate_G(float *data, float *b, float *G, int n_t, int d, int n_tau, int n_i)
{
	unsigned i = threadIdx.x+blockDim.x*blockIdx.x;
		unsigned tau = threadIdx.y+blockDim.y*blockIdx.y;
	if (i >= n_i) return;
	if (tau >= n_tau) return;
		
			float temp1 = 0.0f;
			for (int t = n_tau; t <n_t; t++)
			{
				for (int  p = 0; p < d; p++)
				{
					float temp = 0.0f;
					for (int q = 0; q < d; q++)
					{
						temp = temp + b[p*d*n_tau + q*n_tau + tau]*data[q*n_i*n_t + i*n_t + t-tau];
					}
					temp1 = temp1 + temp*data[p*n_i*n_t + i*n_t + t];
				}
			}	
			G[i*n_tau + tau] = temp1;
		
}

// generate F
__global__ void gpu_generate_F(float *b, float *data, float *F, int d, int n_t, int n_tau, int n_i)
{
	unsigned i = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned tau = threadIdx.y+blockDim.y*blockIdx.y;
	unsigned taub = threadIdx.z+blockDim.z*blockIdx.z;
	if (i >= n_i) return;
	if (tau >= n_tau) return;
	if (taub >= n_tau) return ;
	
			float temp2 = 0.0f;
			for ( int t = n_tau; t < n_t; t++)
			{
				for ( int p = 0; p < d ; p++)
				{
					float temp = 0.0f;
					float temp1 = 0.0f;
					
					for ( int q = 0; q < d ; q++)
					{
						temp = temp + b[p*d*n_tau + q*n_tau + tau]*data[q*n_i*n_t + i*n_t + t - tau];
						temp1 = temp1 + b[p*d*n_tau + q*n_tau + taub]*data[q*n_i*n_t + i*n_t + t - taub];
					}
					temp2 = temp2 +temp*temp1;	
				}
			} 
			F[i*n_tau*n_tau + tau*n_tau + taub] = temp2;
		

}

// generate dell of b
__global__ void gpu_dell_b(float *dell_b, float *b, float *data, float *a, float *H, int d, int n_t, int n_tau, int n_i)
{
	unsigned p = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned qb = threadIdx.y+blockDim.y*blockIdx.y;
	unsigned taub = threadIdx.z+blockDim.z*blockIdx.z;
	if (p >= d) return;
	if (qb >= d) return;
	if (taub >= n_tau) return;

			float temp = 0.0f;
			for (int i = 0; i < n_i; i++)
			{
				for (int t = n_tau; t < n_t; t++)
				{
					temp = temp + data[p*n_i*n_t + i*n_t + t]*a[i*n_tau + taub]*data[qb *n_i*n_t + i*n_t + t-taub];
				}
			}
			float temp1 = 0.0f;
			for (int q = 0; q < d; q++)
			{
				for (int tau = 0; tau < n_tau; tau++)
				{
					//temp1 = temp1 + b[p*d*n_tau + q*n_tau + tau]*H[q*n_tau*d*n_tau + tau*d*n_tau + qb*n_tau + taub];
					temp1 = temp1 + b[p*d*n_tau + q*n_tau + tau]*H[tau*d*n_tau*d + q*n_tau*d + taub*d + qb];
				}
			}
			dell_b[p*d*n_tau + qb*n_tau+ taub] = 2.0f*(temp1 - temp);

}

__global__ void gpu_generate_lip_b(float *lip_b, float *H,int n_tau,int d)
{
	unsigned p = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned qb = threadIdx.y+blockDim.y*blockIdx.y;
	unsigned taub = threadIdx.z+blockDim.z*blockIdx.z;
	float temp = 0.0f;
	if (p >= d) return;
	if (qb >= d) return;
	if (taub >= n_tau) return;
	for (int tau = 0 ; tau <n_tau; tau++)
	{
		temp = H[taub*d*n_tau*d + qb*n_tau*d + tau*d + p]*H[taub*d*n_tau*d + qb*n_tau*d + tau*d + p];
	}
	atomicAdd(&lip_b[0],2.0f*temp);
	//lip_b[0] = temp*2.0f;
}

float *mat_mul(float* A, float* B, int m , int n , int k, int trans)
{
	float *C = new float[m*n];
	for (int i = 0; i < m*n; i++)
	{
		C[i]=0;
	}
	if (trans == 1){
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				for (int l = 0; l < k; l++)
				{
					C[i*n + j] = C[i*n + j] + A[i*k + l]*B[l*n + j];
					//cout << A[i*k + l]*B[l*n + j] << "\n";
				}
			}

		}
	}
	else
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				for (int l = 0; l < k; l++)
				{
					C[i*n + j] = C[i*n + j] + A[i*k + l]*B[j*k + l];
				}
			}
		}
	}
	return C;
}

// calculate the value of the objective function
float calc_obj(float *data, float *a, float *b, float *c, int n_i, int n_k, int n_t, int d, int n_tau, float lambda, float* x_norm)
{
	float objective;

	float val1 = 0.0f;
	for (int k = 0; k < n_k; k++)
	{
		for (int p = 0; p < d; p++)
		{
			val1 =	val1 +  abs(c[k*d + p]) + abs(b[k*d + p]);
		}
	}
	float val2 = 0.0f;
	for (int tau = 0; tau < n_tau; tau++)
	{
		for (int i = 0; i < n_i; i++)
		{
			for (int k = 0; k < n_k; k++)
			{
				val2 = val2 + abs(a[i*n_k*n_tau + k*n_tau + tau]);
			}
		}
	}


	objective  = (x_norm[0]/(n_t*n_i*d)) + lambda*((val1/(n_k*d)) + val2/(n_tau*n_i*n_k));
	return objective;
}

float sign(float temp)
{
	if (temp > 0)
	{
		return 1;
	}
	else if(temp < 0)
	{
		return -1;
	}
	else
	{
		return 0;
	}
}


void initialize_mat(float *init, int size)
{
	for (int i = 0; i < size; i ++)
	{
		init[i] = 0.0f;
	}
}


float total_norm(float *array, int row, int col)
{
	float total = 0.0f;
	for(int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			total = total + abs(array[i*col + j]);
		}
	}
	return total;
}

// function for downsampling
float *downsample_data(float *array, int n_d, int n_i, int n_t, int down_size)
{
	int temp_t = floor(n_t/down_size);
	//cout << "time length1" << temp_t << "\n"; 
	float *downsampled = new float[n_d*n_i*temp_t];
	for (int d = 0; d < n_d; d++)
	{
		for (int i = 0; i < n_i; i++)
		{
			for (int t = 0; t < n_t; t++)
			{
				if (t%down_size == 0)
				{
					downsampled[d*n_i*temp_t + i*temp_t + t/down_size] = array[d*n_i*temp_t + i*temp_t + t];	
				}
			}
		}	
	}
	return downsampled;
}

float *normalize(float* array, int n_d, int n_i, int n_t)
{

	float *normalized = new float[n_d*n_i*n_t];
	for (int d = 0; d < n_d; d++)
	{
		for (int i = 0; i < n_i; i++)
		{
			float mean = 0.0f;
			for (int t = 0; t < n_t; t++)
			{
				mean = mean + array[d*n_i*n_t + i*n_t + t];	
				
			}
			mean = mean/float(n_t);
			float std = 0.0f; 
			for (int t = 0; t < n_t; t++)
			{
				std = std + (array[d*n_i*n_t + i*n_t + t] - mean)*(array[d*n_i*n_t + i*n_t + t] - mean);	
				
			}
			std = sqrt(std); 
			for (int t = 0; t < n_t; t++)
			{
				normalized[d*n_i*n_t + i*n_t + t] = (array[d*n_i*n_t + i*n_t + t] - mean)/std;	
				
			}
		}	
	}
	return normalized;
}


int true_positive(float *origin, float *new_mat, int d, int n_tau, float alpha)
{
	int count = 0;
	for (int i = 0 ; i < d*d*n_tau; i++)
	{
	
		if (new_mat[i]!=0.0f && origin[i]!=0.0f&& origin[i]>alpha && new_mat[i] > alpha)
		{
			count = count + 1;
		}
	}
	return count;
}

int false_positive(float *origin, float *new_mat, int d, int n_tau, float alpha)
{
	int count = 0;
	for (int i = 0 ; i < d*d*n_tau; i++)
	{
		if (new_mat[i]!=0.0f && origin[i]==0.0f && new_mat[i] > alpha)
		{
			count = count + 1;
		}
	}
	return count;
}

int false_negative(float *origin, float *new_mat, int d, int n_tau, float alpha)
{
	int count = 0;
	for (int i = 0 ; i < d*d*n_tau; i++)
	{
		if (new_mat[i]==0.0f && origin[i]!=0.0f&& origin[i]>alpha)
		{
			count = count + 1;
		}
	}
	return count;
}

int true_negative(float *origin, float *new_mat, int d, int n_tau, float alpha)
{
	int count = 0;
	for (int i = 0 ; i < d*d*n_tau; i++)
	{
		if (new_mat[i]==0.0f && origin[i]==0.0f&& new_mat[i] > alpha)
		{
			count = count + 1;
		}
	}
	return count;
}


int main (int argc, char* argv[])
{

	stringstream strValue;
	strValue << argv[1];
	int cuda;
	strValue >> cuda;

	stringstream strValue1;
	strValue1 << argv[2];
	int downsample;
	strValue1 >> downsample;

	stringstream strValue2;
	strValue2 << argv[3];
	float lambda_val;
	strValue2 >> lambda_val;

	cudaSetDevice ( cuda);

	const int n_i = 12;
	const int d = 25;
	//const int n_temp_t = 100;
	const int n_tau = 1;
	//const int downsample = ;
	const int n_t = 1000;
	//const int n_t = 2002/downsample;
	float *test_B = random_B(d, n_tau);
	//float *temp_data = generate_X(origin_B,d,n_i,n_tau,n_t);
	//float* temp_test_data = generate_X(origin_B,d,n_i,n_tau,n_t);

	float* temp_data = new float [d*n_i*n_t];
	float* temp_test_data = new float [d*n_i*n_t];
	for (int i  = 0; i < d*n_i*n_t; i++)
	{
		temp_data[i] = 0;
		temp_test_data[i]=0;	
	}


	// read data

	ifstream in_stream;
  ofstream out_stream;

    int position = 0;
    //Check input and output file.
    in_stream.open("data.dat");
    if (in_stream.fail())
    {
        cout << "Input file opening failed";
        exit(1);
    }

  //array to read data from array as long as we dont reach the end of file marker
	while(! in_stream.eof() && position < d*n_i*n_t)
	{
    in_stream >> temp_data[position];
		//cout << temp_data[position] << "\n";
    position++;
	}
	//temp_data = generate_X(test_B,d,n_i,n_tau,n_t);
	//test_B = random_B(d, n_tau);
	// read B
	ifstream in_stream1;
		cout << "B" << "\n";
    position = 0;
    //Check input and output file.
    in_stream1.open("B.dat");
    if (in_stream1.fail())
    {
        cout << "Input file opening failed";
        exit(1);
    }
	//float* test_B = new float [d*d*n_tau];
  //array to read data from array as long as we dont reach the end of file marker
	while(! in_stream1.eof() && position < d*d*n_tau)
	{
    in_stream1 >> test_B[position];
		cout << test_B[position] << "\n";
    position++;
	}

	float* data = normalize(temp_data,d,n_i,n_t);
	float* test_data = normalize(temp_test_data,d,n_i,n_t);

	float l_val[1] = {lambda_val};

	//float l_val[3] = {0.001f, 10.0f, 100.0f};
	//float nk_val[3] = {50, 50, 50};
	// set lambda and gamma parameter
	float lambda = 0.0f;
	float gamma_a = 2.0f;
	float gamma_b = 3.0f;

	for (int kval = 0; kval < 1; kval++){
		for (int lval = 0; lval < 1 ; lval++){


			// transfer data to device memory
			float *d_data;  (cudaMalloc(&d_data, d*n_i*n_t * sizeof(float)));
			(cudaMemcpy(d_data, data, d*n_i*n_t * sizeof(float), cudaMemcpyHostToDevice));

			float* a = sparse_generate_matrix1(n_i,n_tau,1);
			float* b = generate_matrix3_1(d,d,n_tau);

			float* previous_a = sparse_generate_matrix1(n_i,n_tau,1);
			float* previous_b = generate_matrix3_1(d,d,n_tau);


			float updated_t = 1.0f;
			float updated_t1 = 0.0f;

			float rel_error = 0.0f;

			int loop = 1;

			// calculate norm of data
			float norm_data = 0.00f;
			for (int i = 0; i < n_i; i++)
			{
				for (int p = 0; p < d; p++)
				{		
					//cout << data[p*n_i*n_t + i*n_t + 0] << "\n"; 						
					for (int t = n_tau; t < n_t; t++)
					{
						norm_data = norm_data + ((data[p*n_i*n_t + i*n_t + t])*(data[p*n_i*n_t + i*n_t + t]));
					}
				}
			}

			cout << "norm _ value:" << norm_data << "\n";

			// open file stream
			ostringstream ss1;
			ss1 << downsample;
			string str1 = ss1.str();
			ostringstream ss;
			ss << l_val[lval];
			string str = ss.str();
			string file =  "result_"+str1+"/output"+str+".csv";
			ofstream myfile;
			myfile.open(file.c_str());
			myfile.close();

			lambda = l_val[lval];

			cout << "lambda" << lambda << "\n";
			float pre_error = 10.0f;
			float new_error = 6.0f;
			while (loop < 80){
			//while((pre_error-new_error)/pre_error > 0.01 ){
				cout << "loop number" << loop << "\n";
				cout << "change" << (pre_error-new_error)/pre_error << "\n";
				//cout << "norm of b" << total_norm(b,d*d,n_tau);
				//cout << "norm of a" << total_norm(a,n_i,n_tau);
				//cout << "change: " << change << "\n";
				//cout << "n_k value" << n_k << "\n";

				clock_t tStart = clock();
				// Extrapolation step
				updated_t1 = (1 + sqrt((1 + 4*(updated_t*updated_t))))/2;
				//cout << "updated t value : " << updated_t << "\n";

				// transfer a to device memory
				float* d_a; (cudaMalloc(&d_a, n_i*n_tau * sizeof(float)));
				(cudaMemcpy(d_a, a, n_i*n_tau * sizeof(float), cudaMemcpyHostToDevice));

				// transfer b to device memory
				float* d_b; (cudaMalloc(&d_b, d*d*n_tau * sizeof(float)));
				(cudaMemcpy(d_b, b, d*d*n_tau* sizeof(float), cudaMemcpyHostToDevice));

				// Starting derivate of a
				//-----------------------------------------------------------------------------------------//

				//cout << "Starting F \n" ;
				// get the F matrix
				float *d_F;  (cudaMalloc(&d_F, n_i*n_tau*n_tau* sizeof(float)));
				float *F = new float[n_i*n_tau*n_tau];
				for (int i = 0 ; i < n_i*n_tau*n_tau; i++)
				{
					F[i] = 0.0f;

				}

				(cudaMemcpy(d_F, F, n_i*n_tau*n_tau* sizeof(float), cudaMemcpyHostToDevice));
				dim3 block(8,1,1);
				dim3 grid((n_i+block.x-1)/block.x,(n_tau+block.y-1)/block.y,(n_tau+block.z-1)/block.z);
				cudaDeviceSynchronize();
				gpu_generate_F<<<grid,block>>>(d_b, d_data, d_F, d, n_t,n_tau,n_i);
				cudaDeviceSynchronize();
				(cudaMemcpy(F, d_F, n_i*n_tau*n_tau* sizeof(float), cudaMemcpyDeviceToHost));
				//cout << "Starting G \n" ;
				// get the G matrix
				float *d_G;  (cudaMalloc(&d_G, n_i*n_tau* sizeof(float)));
				float *G = new float[n_i*n_tau];
				for (int i = 0 ; i < n_i*n_tau; i++)
				{
					G[i] = 0.0f;

				}

				(cudaMemcpy(d_G, G, n_i*n_tau* sizeof(float), cudaMemcpyHostToDevice));
				dim3 blockG(8,1);
				dim3 gridG((n_i+blockG.x-1)/blockG.x,(n_tau+blockG.y-1)/blockG.y);
				cudaDeviceSynchronize();
				gpu_generate_G<<<gridG,blockG>>>(d_data, d_b, d_G, n_t, d, n_tau, n_i);
				cudaDeviceSynchronize();
				(cudaMemcpy(G, d_G, n_i*n_tau* sizeof(float), cudaMemcpyDeviceToHost));

				// lipschitz limits for a
				//cout << "before lip of a completed" << "\n" ;
				//cout << "norm of F" << total_norm(F,n_i*n_tau, n_tau) << "\n";
				//cout << "norm of G" << total_norm(G,n_i, n_tau) << "\n";
				float lip1 = 0.0f;
				float* lip_a = new float[n_i];
				for (int k = 0; k < n_i; k++)
				{
					lip_a[k] = 0.0f;
					float *F_temp = new float[n_tau*n_tau];
					for (int i = 0; i < n_tau*n_tau; i++)
					{
						F_temp[i] = F[k*n_tau*n_tau + i];
					}
					for (int i = 0; i < 3; i++)
					{
						float* test = sparse_generate_matrix1(n_tau,1,1);
						lip1 = power_iter(F_temp,test,n_tau,20);
						if (lip_a[k] < lip1)
						{
							lip_a[k] = lip1;
						}
						delete[] test;

					}
					lip_a[k] = lip_a[k]*(2.0f);
					delete[] F_temp;
				}
				//cout << "Lip of a completed" << "\n" ;
				// calculate gradient for a
				float *dell_a = new float[n_i*n_tau];
				for(int i = 0; i < n_i*n_tau; i++)
				{
					dell_a[i] = 0.0f;
				}

				float *d_dell_a;  (cudaMalloc(&d_dell_a, n_i*n_tau* sizeof(float)));
				(cudaMemcpy(d_dell_a, dell_a, n_i*n_tau* sizeof(float), cudaMemcpyHostToDevice));
				dim3 blocka(8);
				dim3 grida((n_i+blocka.x-1)/blocka.x);
				cudaDeviceSynchronize();
				gpu_dell_a<<<grida,blocka>>>(d_G, d_a, d_F, d_dell_a, n_i, n_tau);
				cudaDeviceSynchronize();
				(cudaMemcpy(dell_a, d_dell_a, n_i*n_tau* sizeof(float), cudaMemcpyDeviceToHost));

				// update a
				float denom1 = 0.0f;
				float *updated_a = new float[n_i*n_tau];
				for (int i = 0; i < n_i; i++)
				{
					//cout << "lip of a" << lip_a[i] << "\n";
					//cout << "derivae of a" << dell_a[i*2 + 2] << "\n";
					for (int tau = 0; tau < n_tau; tau++)
					{
						//updated_a[i*n_tau + tau] = a[i*n_tau + tau] - (1.0f/(gamma_a*lip_a[i]))*dell_a[i*n_tau + tau] -  (float(0.001f)*float(d*n_i)/(gamma_a*lip_a[i]*float(n_i*n_tau)))*a[i*n_tau + tau] ;
						a[i*n_tau + tau] = a[i*n_tau + tau] - (1.0f/(gamma_a*lip_a[i]))*dell_a[i*n_tau + tau] ;
						//updated_a[i*n_tau + tau] = a[i*n_tau + tau];
						denom1 = denom1 + a[i*n_tau + tau];
					}
				}

				float tempo1 = 0.0f;
				for (int i = 0; i < n_i; i++)
				{
					tempo1 = 1.0f- (((float)(lambda)*float(d*n_i))/((float(gamma_a*lip_a[i]))*float(n_i*n_tau)*sqrt(denom1)));
					for (int tau = 0; tau < n_tau; tau++)
					{
							
						if (tempo1 > 0.0f)
						{
								updated_a[i*n_tau + tau] = a[i*n_tau + tau]*tempo1 ;
						}
						else
						{
							updated_a[i*n_tau + tau] = 0.0f;
						}						
					}
				}


				// update a using extrapolation
				for (int i = 0; i < n_i; i++)
				{

					for (int tau = 0; tau < n_tau; tau++)
					{
						//a[i*n_tau + tau] = updated_a[i*n_tau + tau];

						a[i*n_tau + tau] = updated_a[i*n_tau + tau] + ((updated_t - 1)/updated_t1)*(updated_a[i*n_tau + tau] - previous_a[i*n_tau + tau]) ;
						//a[i*n_tau + tau] = updated_a[i*n_tau + tau] ;

					}
				}
								cout << "norm of a" << total_norm(a,n_i,n_tau) << "\n" ;
				cudaFree(d_dell_a);
				delete[] dell_a;
				delete[] lip_a;
				cudaFree(d_G);
				delete[] G;
				cudaFree(d_F);
				delete[] F;

				//cout << "derivate of a completed" << "\n" ;
				//cout << "norm of a" << total_norm(a,n_i,n_tau) << "\n";
				// Starting derivate of b
				//----------------------------------------------------------------------------------------------------//

				float* d_a2; (cudaMalloc(&d_a2, n_i*n_tau * sizeof(float)));
				(cudaMemcpy(d_a2, updated_a, n_i*n_tau * sizeof(float), cudaMemcpyHostToDevice));
				// get the H matrix
				float *d_H;  (cudaMalloc(&d_H, n_tau*d*n_tau*d* sizeof(float)));
				float *H = new float[n_tau*d*n_tau*d];
				for (int i = 0 ; i < n_tau*d*n_tau*d; i++)
				{
					H[i] = 0.0f;

				}
				CudaCheckError();
				(cudaMemcpy(d_H, H, n_tau*d*n_tau*d* sizeof(float), cudaMemcpyHostToDevice));
				dim3 blockH(1,8,8);
				dim3 gridH((n_tau+blockH.x-1)/blockH.x,(d+blockH.y-1)/blockH.y,(d+blockH.z-1)/blockH.z);
				cudaDeviceSynchronize();
				gpu_generate_H<<<gridH,blockH>>>(d_a2, d_data, d_H, n_t, n_i, n_tau, d);
				cudaDeviceSynchronize();
				(cudaMemcpy(H, d_H, n_tau*d*n_tau*d*sizeof(float), cudaMemcpyDeviceToHost));

				
				// lipschitz limits for b
				float lip2 = 0.0f;
				float lip_b1 = 0.0f;
				float *H_temp = new float[n_tau*d*n_tau*d];
				for (int i = 0; i < n_tau*d*n_tau*d; i++)
				{
					
						H_temp[i] = H[i];
					

				}
				for (int i = 0; i < 4; i++)
				{
					float* test = sparse_generate_matrix1(n_tau*d,1,1);
					lip2 = power_iter(H_temp,test,n_tau*d,30);
					if (lip_b1 < lip2)
					{
						lip_b1 = lip2;
					}
					delete[] test;
				}
				lip_b1 = lip_b1*(2.0f/float(d*n_i));
				cout << "lip of b" << lip_b1 << "\n";

				delete[] H_temp;
				
				// calculate the derivate of b
/*
				float *d_lip_b;  (cudaMalloc(&d_lip_b, 1* sizeof(float)));
				float *lip_b = new float[1];
				lip_b[0] = 0.0f;
				CudaCheckError();
				(cudaMemcpy(d_lip_b, lip_b, 1* sizeof(float), cudaMemcpyHostToDevice));
				dim3 block_lip_b(2,16,16);
				dim3 grid_lip_b((n_tau+block_lip_b.x-1)/block_lip_b.x,(d+block_lip_b.y-1)/block_lip_b.y,(d+block_lip_b.z-1)/block_lip_b.z);
				cudaDeviceSynchronize();
				gpu_generate_lip_b<<<grid_lip_b,block_lip_b>>>(d_lip_b, d_H, n_tau, d);
				cudaDeviceSynchronize();
				(cudaMemcpy(lip_b, d_lip_b, 1*sizeof(float), cudaMemcpyDeviceToHost));
*/

				float *d_dell_b;  (cudaMalloc(&d_dell_b, d*d*n_tau* sizeof(float)));
				float *dell_b = new float[d*d*n_tau];
				for (int i = 0 ; i < d*d*n_tau; i++)
				{
					dell_b[i] = 0.0f;

				}
				//cout << "lip of b" << lip_b[0] << "\n";
				(cudaMemcpy(d_dell_b, dell_b, d*d*n_tau* sizeof(float), cudaMemcpyHostToDevice));
				dim3 blockb(8,8,1);
				dim3 gridb((d+blockb.x-1)/blockb.x,(d+blockb.y-1)/blockb.y,(n_tau+blockb.z-1)/blockb.z);
				cudaDeviceSynchronize();
				gpu_dell_b<<<gridb,blockb>>>(d_dell_b,d_b, d_data, d_a2, d_H, d, n_t,n_tau,n_i);
				cudaDeviceSynchronize();
				(cudaMemcpy(dell_b, d_dell_b, d*d*n_tau*sizeof(float), cudaMemcpyDeviceToHost));

				float denom = 0.0f;
				float tempo = 0.0f;
				float tempo2 = 0.0f;
				// Soft thresholding and update b
				float *updated_b = new float[d*d*n_tau];
				
				for (int p = 0; p < d; p++)
				{
					for (int q = 0; q < d ; q++)	
					{
						denom = 0.0f;
						for (int tau = 0; tau < n_tau; tau ++)					
						{
							b[p*d*n_tau + q*n_tau + tau] = b[p*d*n_tau + q*n_tau + tau] - (1.0f/(gamma_b*lip_b1))*dell_b[p*d*n_tau + q*n_tau + tau]*(1.0f/float(d*n_i));
							denom = denom + b[p*d*n_tau + q*n_tau + tau]*b[p*d*n_tau + q*n_tau + tau];
						}
						tempo = 1.0f- (((float)(lambda))/((float(gamma_b*lip_b1))*float(d*d*n_tau)*sqrt(denom)));
						//tempo1 = 1.0f+ (((float)(lambda))/(((float)(gamma_b*lip_b1))*sqrt(denom)*float(d*d*n_tau)));
						//cout << "tempo value" << tempo << "\n";
						for (int tau = 0; tau < n_tau; tau ++)						
						{
							//cout << "hello" << "\n";
							if (tempo > 0.0f )
							{
								updated_b[p*d*n_tau + q*n_tau + tau] = b[p*d*n_tau + q*n_tau + tau]*tempo;
							}
							
							else
							{
								updated_b[p*d*n_tau + q*n_tau + tau] = 0.0f;
							}
						}
					}
				}
/*
				for (int p = 0; p < d; p++)
				{
					for (int q = 0; q < d ; q++)	
					{

						for (int tau = 0; tau < n_tau; tau ++)					
						{
							b[p*d*n_tau + q*n_tau + tau] = b[p*d*n_tau + q*n_tau + tau] - (1.0f/(gamma_b*lip_b1))*dell_b[p*d*n_tau + q*n_tau + tau]*(1.0f/float(d*n_i));

						
							//cout << "hello" << "\n";
							if (b[p*d*n_tau + q*n_tau + tau] >= (((float)(lambda))/((float(gamma_b*lip_b1))*float(d*d*n_tau))) )
							{
								updated_b[p*d*n_tau + q*n_tau + tau] = b[p*d*n_tau + q*n_tau + tau]-(((float)(lambda))/((float(gamma_b*lip_b1))*float(d*d*n_tau))) ;
							}
							else if (b[p*d*n_tau + q*n_tau + tau] <= -(((float)(lambda))/((float(gamma_b*lip_b1))*float(d*d*n_tau))) )
							{
								updated_b[p*d*n_tau + q*n_tau + tau] = b[p*d*n_tau + q*n_tau + tau](((float)(lambda))/((float(gamma_b*lip_b1))*float(d*d*n_tau))) ;
							}
							else
							{
								updated_b[p*d*n_tau + q*n_tau + tau] = 0.0f;
							}
						}
					}
				}
*/
				//delete[] lip_b;
				//cudaFree(d_lip_b);
				cudaFree(d_H);
				cudaFree(d_dell_b);
				// update b using extrapolation
				for (int p = 0; p < d; p++)
				{
					for (int q = 0; q < d ; q++)
					{
						for (int tau = 0; tau < n_tau; tau ++)
						{
							//b[p*d*n_tau + q*n_tau + tau] = updated_b[p*d*n_tau + q*n_tau + tau];
							b[p*d*n_tau + q*n_tau + tau] = updated_b[p*d*n_tau + q*n_tau + tau] + ((updated_t - 1)/updated_t1)*(updated_b[p*d*n_tau + q*n_tau + tau] - previous_b[p*d*n_tau + q*n_tau + tau]);

						}
					}
				}
				delete[] dell_b; 
				delete[] H;
				cudaFree(d_a2);
				cout << "norm of b" << total_norm(b,d*d,n_tau) << "\n" ;

				cudaFree(d_a);
				cudaFree(d_b);
				

				//----------------------------------------------------------------------------------------------------//
				// store the values of updated a
				for (int i = 0; i < n_i; i++)
				{
					for (int tau = 0; tau < n_tau; tau++)
					{
						previous_a[i*n_tau + tau] = updated_a[i*n_tau + tau];
					}
				}
				int nnz = 0;
				// store the values of updated b
				for (int p = 0; p < d; p++)
				{
					for (int q = 0; q < d ; q++)
					{
						for (int tau = 0; tau < n_tau; tau ++)
						{
							previous_b[p*d*n_tau + q*n_tau + tau] = updated_b[p*d*n_tau + q*n_tau + tau];
							if (b[p*d*n_tau + q*n_tau + tau] != 0)
							{
								nnz = nnz + 1;
							}
						}
					}
				}
				cout << "number of non zeros" << nnz << "\n";
				float *d_a1; (cudaMalloc(&d_a1, n_i*n_tau * sizeof(float)));
				(cudaMemcpy(d_a1, a, n_i*n_tau * sizeof(float), cudaMemcpyHostToDevice));

				// transfer b to device memory
				float *d_b1; (cudaMalloc(&d_b1, d*d*n_tau * sizeof(float)));
				(cudaMemcpy(d_b1, b, d*d*n_tau* sizeof(float), cudaMemcpyHostToDevice));

				loop = loop + 1;
				updated_t = updated_t1;
				delete[] updated_a;
				delete[] updated_b;


				printf("%.2f\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
				float *x_norm = new float[1];
				x_norm[0] = 0.0f;
				float *d_x_norm; cudaMalloc(&d_x_norm, 1*sizeof(float));
				cudaMemcpy(d_x_norm,x_norm,1*sizeof(float) , cudaMemcpyHostToDevice);

				dim3 block_norm(8);
				dim3 grid_norm((d+block_norm.x-1)/block_norm.x);
				indvi_norms1<<<grid_norm,block_norm>>>(d_data,d_a1, d_b1, n_i, n_t, d, n_tau, d_x_norm);
				cudaDeviceSynchronize();
				cudaMemcpy(x_norm,d_x_norm,1*sizeof(float) , cudaMemcpyDeviceToHost);

				rel_error = x_norm[0]/norm_data;
				pre_error = new_error;
				new_error = rel_error;
				//rel_error = 1-rel_error;
				//cout << "train error : " << rel_error << "\n";
				cout << rel_error << "\n";

				cudaFree(d_a1);
				cudaFree(d_b1);
				delete[] x_norm;
				cudaFree(d_x_norm);

			}
			
			ofstream myfilea;
			myfilea.open(("a_result"+str+".csv").c_str());
			for (int i = 0; i < n_i; i++)
			{
					for (int tau = 0; tau < n_tau; tau++)
					{
						myfilea << a[i*n_tau + tau] << ",";

					}
					myfilea << "\n";
				

			}
			myfilea.close();
			ofstream myfileb;
			myfileb.open (("b_result"+str+".csv").c_str());
			for (int q = 0; q < d; q++)
			{
				for (int p = 0; p < d; p++)
				{
					for (int tau = 0; tau < n_tau; tau++)
					{
						myfileb << b[q*d*n_tau + p*n_tau + tau] << ",";
					}				
				}
				myfileb << "\n";
			}
			int nnz = 0;
			for (int i = 0; i < d*d*n_tau ; i++){

				if (test_B[i] != 0)
				{
					nnz = nnz + 1;
				}			
			}
			myfileb.close();
			float alpha = 0.00f;
			cout << "nnz in test_B" << nnz;
			cout << "true positive" << float(true_positive(test_B, b, d, n_tau,alpha));
			cout << "sensitivity:" << float(true_positive(test_B, b, d, n_tau,alpha))/float(false_negative(test_B, b, d, n_tau,alpha)+true_positive(test_B, b, d, n_tau,alpha));
			cout << "precision:" << float(true_positive(test_B, b, d, n_tau,alpha))/float(false_positive(test_B, b, d, n_tau,alpha)+true_positive(test_B, b, d, n_tau,alpha));
			cout << "accuracy:" << float(true_positive(test_B, b, d, n_tau,alpha)+true_negative(test_B, b, d, n_tau,alpha))/float(false_positive(test_B, b, d, n_tau,alpha)+true_positive(test_B, b, d, n_tau,alpha)+true_negative(test_B, b, d, n_tau,alpha)+false_negative(test_B, b, d, n_tau,alpha));
cout << "F1:" << float(2.0f*true_positive(test_B, b, d, n_tau,alpha))/float(false_positive(test_B, b, d, n_tau,alpha)+2.0f*true_positive(test_B, b, d, n_tau,alpha)+false_negative(test_B, b, d, n_tau,alpha));
			
			cudaFree(d_data);
			// transfer a to device memory
			float* d_a; (cudaMalloc(&d_a, n_i*n_tau * sizeof(float)));
			(cudaMemcpy(d_a, a, n_i*n_tau * sizeof(float), cudaMemcpyHostToDevice));

			// transfer b to device memory
			float* d_b; (cudaMalloc(&d_b, d*d*n_tau * sizeof(float)));
			(cudaMemcpy(d_b, b, d*d*n_tau* sizeof(float), cudaMemcpyHostToDevice));

			// transfer data to device memory
			(cudaMalloc(&d_data, d*n_i*n_t * sizeof(float)));
			(cudaMemcpy(d_data, test_data, d*n_i*n_t * sizeof(float), cudaMemcpyHostToDevice));

			// calculate norm of data
			norm_data = 0.00f;
			for (int i = 0; i < n_i; i++)
			{
				for (int t = n_tau; t < n_t; t++)
				{
					for (int p = 0; p < d; p++)
					{
						norm_data = norm_data + ((test_data[p*n_i*n_t + i*n_t + t])*(test_data[p*n_i*n_t + i*n_t + t]));
					}
				}
			}

			//cout << "norm _ value:" << norm_data << "\n";
			float *x_norm = new float[1];
			x_norm[0] = 0.0f;
			float *d_x_norm;	cudaMalloc(&d_x_norm, 1*sizeof(float));
			cudaMemcpy(d_x_norm,x_norm,1*sizeof(float) , cudaMemcpyHostToDevice);


				dim3 block_norm(16);
				dim3 grid_norm((d+block_norm.x-1)/block_norm.x);
				indvi_norms1<<<grid_norm,block_norm>>>(d_data,d_a, d_b, n_i, n_t, d, n_tau, d_x_norm);
				cudaDeviceSynchronize();
				cudaMemcpy(x_norm,d_x_norm,1*sizeof(float) , cudaMemcpyDeviceToHost);

				rel_error = x_norm[0]/norm_data;

				//rel_error = 1-rel_error;
				//cout << "train error : " << rel_error << "\n";
				cout << rel_error << "\n";

				cudaFree(d_a);
				cudaFree(d_b);
				delete[] x_norm;
				cudaFree(d_x_norm);
			


		}
	}
}


