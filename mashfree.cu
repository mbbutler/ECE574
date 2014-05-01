// Forrest Flagg
// Brady Butler

#include <cmath>
#include <string.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define GRAV	-9.8
#define K_CON	1
#define RHO		1000
#define MU		1
#define NPPB	1024		// number of particles per block

using namespace std;

typedef float p_type;

typedef struct Particle_s Particle;
typedef struct Force_s Force;

struct Particle_s {
	p_type x,y,z;
	p_type vx, vy, vz;
	p_type fx, fy, fz;
	p_type ofx, ofy, ofz;
	p_type den;
	p_type mass;
};

struct Force_s {
	p_type x;
	p_type y;
	p_type z;
};

void init_particles(Particle* part, int n, dim3 bn);
__global__ void run(Particle *data, p_type dt, p_type h);
__device__ void move(Particle *local, p_type dt);
__device__ void boundary_conditions(Particle *local, float3 max, float3 min);
__device__ void mem_s2g(Particle *data, Particle *local);
__device__ Particle *mem_g2s(Particle *data);
__device__ Particle *calc_all_forces(Particle *data, Particle *local, p_type h);
__device__ void calc_den(Particle *data, Particle *local, p_type h);
__device__ p_type calc_l_pf(Particle *loc, Particle *bloc, p_type h);
__device__ Force calc_l_visc(Particle *loc, Particle *bloc, p_type h);
__device__ p_type calc_l_den(Particle *loc, Particle *bloc, p_type h);
__device__ void calc_den(Particle *data, Particle *local, p_type h);
__device__ p_type calc_l_pf(Particle *loc, Particle *bloc, p_type h);

using namespace std;

int main(int argc, char *argv[]) {

	if (argc != 2) {
		cout << "Error:  Incorrect arguments." << endl;
		cout << "Usage:  " << argv[0] << " numParticles " << endl;
		exit(-1);
	}

	// Get number of particles from input
	string val_string = argv[1];
	int N = atoi(val_string.c_str());
	// Setup host array for position and velocity info
	Particle *part;
	size_t size = N*sizeof(Particle);
	cout << "Allocating for particles on host" << endl;
	part = (Particle*)malloc(size);
	if (!part) {
			cout << "Malloc failed" << endl;
			exit(0);
	}
	cout << "Done allocating for particles on host" << endl;

	dim3 threadsPerBlock(NPPB);
	dim3 numBlocks(3, 3, 3);

	init_particles(part, NPPB, numBlocks);

	// Setup device array for pos and vel info
	Particle *d_part;

	// Allocate memory on device
	cout << "Allocating for particles on device" << endl;
	cudaError_t err  = cudaMalloc(&d_part, size);
	if (err != cudaSuccess) {
		cout << "cudaMalloc failed" << endl;
		exit(0);
	}
	cout << "Done allocating for particles on device" << endl;

	// Copy memory to device
	cout << "Copying from host to device" << endl;
	err = cudaMemcpy(d_part, part, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "cudaMemcpy failed with error: " << err << endl;
		exit(0);
	}
	cout << "Done copying from host to device" << endl;

	p_type h, dt;
	dt = 0.01;
	h = 0.1;
//	<<<numBlocks, threadsPerBlock, 2*NPPB*sizeof(Particle)>>>
	run<<<numBlocks, threadsPerBlock>>>(d_part, dt, h);

	// Copy memory from  device to host
	cout << "Copying from device to host" << endl;
	err = cudaMemcpy(part, d_part, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { 
		cout << "cudaMemcpy failed with error: " << err << endl;
		exit(0);
	}
	cout << "Done copying from device to host" << endl;
}

void init_particles(Particle* part, int n, dim3 bn) {
	for (int z = 0; z < bn.z; ++z) {
		for (int y = 0; y < bn.y; ++y) {
			for (int x = 0; x < bn.x; ++x) {
				int id = bn.z*bn.y*bn.x*z + bn.y*bn.x*y + bn.x*x;
				for (int i = 0; i < n; ++i) {
					part[i].x = id + i;
					part[i].y = id + i;
					part[i].z = id + i;
					part[i].vx = 0; 
					part[i].vy = 0;
					part[i].vz = 0;
				}
			}
		}
	}
}

__global__ void run(Particle *data, p_type dt, p_type h) {
	Particle *local = mem_g2s(data);
	calc_den(data, local, h);
	calc_all_forces(data, local, h);
	move(local, dt);
	mem_s2g(data, local);
}

__device__ void move(Particle *local, p_type dt) {
	Particle *me = &local[threadIdx.x];
	me->x += me->vx*dt;
	me->y += me->vy*dt;
	me->z += me->vz*dt;
	me->vx += (me->fx + me->ofx)/2*dt;
	me->vy += (me->fy + me->ofy)/2*dt;
	me->vz += (me->fz + me->ofz)/2*dt;
}

__device__ void boundary_conditions(Particle *local, float3 max, float3 min) {
	Particle *me = &local[threadIdx.x];
	dim3 val;
	val.x = me->x > max.x || me->x < min.x; 
	val.y = me->x > max.y || me->y < min.y; 
	val.z = me->x > max.z || me->z < min.z; 
	me->vx += -1.9 * me->vx * val.x;
	me->vy += -1.9 * me->vy * val.y;
	me->vz += -1.9 * me->vz * val.z;
}

__device__ void mem_s2g(Particle *data, Particle *local) {
	int off = blockIdx.z*blockDim.z*blockDim.y*blockDim.x + blockIdx.y*blockDim.y*blockDim.x + blockIdx.x*blockDim.x;
	data[off + threadIdx.x] = local[threadIdx.x];
	__syncthreads();
}

__device__ Particle *mem_g2s(Particle *data) {
	__shared__ Particle local[NPPB];
	int off = blockIdx.z*blockDim.z*blockDim.y*blockDim.x + blockIdx.y*blockDim.y*blockDim.x + blockIdx.x*blockDim.x;
	local[threadIdx.x] = data[off + threadIdx.x];
	__syncthreads();
	return local;
}

__device__ Particle *calc_all_forces(Particle *data, Particle *local, p_type h) {
	p_type f_press = 0;
	Force force, ret_force;
	force.x = force.y = force.z = 0;
	int x, y, z;
	for (int i = -1; i < 2; ++i) {
		for (int j = -1; j < 2; ++j) {
			for (int k = -1; k < 2; ++k) {
				z = blockIdx.z + i;
				y = blockIdx.y + j;
				x = blockIdx.x + k;

				if (x<0 || x > gridDim.x - 1) {
					continue;
				}
				if (y<0 || y > gridDim.y - 1) {
					continue;
				}
				if (z<0 || z > gridDim.z - 1) {
					continue;
				}

				int id = z*blockDim.z*blockDim.y*blockDim.x + y*blockDim.y*blockDim.x + x*blockDim.x;
				__shared__ Particle shared[NPPB];
				shared[threadIdx.x] = data[id + threadIdx.x];
				__syncthreads();

				f_press += calc_l_pf(local, shared, h);
				ret_force = calc_l_visc(local, shared, h);
				force.x += ret_force.x;
				force.y += ret_force.y;
				force.z += ret_force.z;
			}
		}
	}
	__syncthreads();

	local[threadIdx.x].ofx = local[threadIdx.x].fx;
	local[threadIdx.x].ofy = local[threadIdx.x].fy;
	local[threadIdx.x].ofz = local[threadIdx.x].fz;

	f_press = -f_press/(2*local[threadIdx.x].den);
	local[threadIdx.x].fx = f_press + MU/local[threadIdx.x].den*force.x;
	local[threadIdx.x].fy = f_press + MU/local[threadIdx.x].den*force.y;
	local[threadIdx.x].fz = f_press + MU/local[threadIdx.x].den*force.z + local[threadIdx.x].mass*GRAV;

	return local;
}


__device__ Force calc_l_visc(Particle *loc, Particle *bloc, p_type h) {
	Force force;
	force.x = force.y = force.z = 0;
	int id = threadIdx.x;
	Particle* me = &loc[id];
	Particle* you;
	p_type con;
	p_type h6 = pow(h,6);
	for(int i = 0; i < blockDim.x; ++i) {
		you = &bloc[i];
		p_type r = sqrt((me->x - you->x)*(me->x - you->x) + (me->y - you->y)*(me->y - you->y) + (me->z - you->z)*(me->z - you->z));
		con = (r>=0 || r<=h) * you->mass/you->den*45/(M_PI*h6);
		force.x += (you->vx-me->vx) * con;
		force.y += (you->vy-me->vy) * con;
		force.z += (you->vz-me->vz) * con;
	}
	return force;
}

__device__ p_type calc_l_pf(Particle *loc, Particle *bloc, p_type h) {
	int id = threadIdx.x;
	p_type f = 0;
	Particle me = loc[id];
	Particle you;
	p_type h6 = pow(h,6);
	for(int i = 0; i < blockDim.x; ++i) {
		you = bloc[i];
		p_type r = sqrt((me.x - you.x)*(me.x - you.x) + (me.y - you.y)*(me.y - you.y) + (me.z - you.z)*(me.z - you.z));
		f += (r>=0 || r<=h) * me.mass * K_CON * (2*RHO + me.den + you.den) * 15/(M_PI*h6)*pow(h-r, 3);
	}
	return f;
}

// Calculates density of particle and stores in the struct
__device__ void calc_den(Particle *data, Particle *local, p_type h) {
	p_type den = 0;
	int x, y, z;
	for (int i = -1; i < 2; ++i) {
		for (int j = -1; j < 2; ++j) {
			for (int k = -1; k < 2; ++k) {
				z = blockIdx.z + i;
				y = blockIdx.y + j;
				x = blockIdx.x + k;
				if (x<0 || x > gridDim.x - 1) {
					continue;
				}
				if (y<0 || y > gridDim.y - 1) {
					continue;
				}
				if (z<0 || z > gridDim.z - 1) {
					continue;
				}
				int id = z*blockDim.z*blockDim.y*blockDim.x + y*blockDim.y*blockDim.x + x*blockDim.x;
				__shared__ Particle shared[NPPB];
				shared[threadIdx.x] = data[id + threadIdx.x];
				__syncthreads();
				den += calc_l_den(local, shared, h);
			}
		}
	}
	local[threadIdx.x].den = den;
}

__device__ p_type calc_l_den(Particle *loc, Particle *bloc, p_type h) {
	p_type den = 0;
	int id = threadIdx.x;
	for(int i = 0; i < blockDim.x; ++i) {
		p_type rsq = (loc[id].x - bloc[i].x)*(loc[id].x - bloc[i].x) + (loc[id].y - bloc[i].y)*(loc[id].y - bloc[i].y) + (loc[id].z - bloc[i].z)*(loc[id].z - bloc[i].z);
		den += (rsq >= 0 || rsq <= h*h) * loc[id].mass * 315*(h*h-rsq)/(64*M_PI*pow(h,9));
	}
	return den;
}

