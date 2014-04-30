#include <cmath>
#include <string.h>
#include <cstring>
#include <iostream>

#define GRAV -9.8
#define K_CON 1
#define RHO 1
#define MU 1

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

int main(int argc, char *argv[]) {
	// Get number of particles from input
	string val_string = argv[1];
	int N = atoi(val_string.c_str());
	// Setup host array for position and velocity info
	Particle *part;
	size_t size = N*sizeof(Particle);
	part= malloc(size);
	if (!part) {
			cout << "Malloc failed" << endl;
			exit(0);
	}
	init_particles(particles);

	// Setup device array for pos and vel info
	Particle *d_part;

	// Allocate memory on device
	cudaError_t err  = cudaMalloc(&d_part, size);
	if (err != cudaSuccess) {
		cout << "cudaMalloc failed" << endl;
		exit(0);
	}
	// Copy memory to device
	err = cudaMemcpy(d_part, part, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "cudaMemcpy failed with error: " << err << endl;
		exit(0);
	}

	dim3 threadsPerBlock(1024);
	dim3 numBlocks(3, 3, 3);

	<<<numBlocks, threadsPerBlock, 2*1024*sizeof(Particle)>>>
}

__global__ void run(Particle *data, p_type dt, p_type h) {
	Particle *local = mem_g2s(data);
	calc_den(data, local, h);
	calc_all_forces(data, local, h);
	move(local, dt);
	mem_s2g(data, local);
}

__device__ void move(Particle *local, p_type dt) {
	Particle *me = &local[ThreadIdx.x];
	me->x += me->vx*dt;
	me->y += me->vy*dt;
	me->z += me->vz*dt;
	me->vx += (me->fx + me->ofx)/2*dt;
	me->vy += (me->fy + me->ofy)/2*dt;
	me->vz += (me->fz + me->ofz)/2*dt;
}

__device__ void *mem_s2g(Particle *data, Particle *local) {
	int off = BlockIdx.z*BlockDim.z*BlockDim.y*BlockDim.x + BlockIdx.y*BlockDim.y*BlockDim.x + BlockIdx.x*BlockDim.x;
	data[off + ThreadIdx.x] = local[ThreadIdx.x];
	__syncthreads();
}

__device__ Particle *mem_g2s(Particle *data) {
	__shared__ Particle local[1024];
	int off = BlockIdx.z*BlockDim.z*BlockDim.y*BlockDim.x + BlockIdx.y*BlockDim.y*BlockDim.x + BlockIdx.x*BlockDim.x;
	local[ThreadIdx.x] = data[off + ThreadIdx.x];
	__syncthreads();
	return local;
}

__device__ Particle *calc_all_forces(Particle *data, Particle *local, p_type h) {
	p_type f_press = 0;
	Force force, ret_force;
	force.x = force.y = force.z = 0;
	int x, y, z;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				z = BlockIdx.z + i;
				y = BlockIdx.y + j;
				x = BlockIdx.x + k;

				if (x<0 || x > GridDim.x - 1) {
					continue;
				}
				if (y<0 || y > GridDim.y - 1) {
					continue;
				}
				if (z<0 || z > GridDim.z - 1) {
					continue;
				}

				int id = z*BlockDim.z*BlockDim.y*BlockDim.x + y*BlockDim.y*BlockDim.x + x*BlockDim.x;
				__shared__ Particle shared[1024];
				shared[ThreadIdx.x] = data[id + ThreadIdx.x];
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

	local[ThreadIdx.x].ofx = local[ThreadIdx.x].fx;
	local[ThreadIdx.x].ofy = local[ThreadIdx.x].fy;
	local[ThreadIdx.x].ofz = local[ThreadIdx.x].fz;

	f_press = -f_press/(2*local[ThreadIdx.x].den);
	local[ThreadIdx.x].fx = f_press + MU/local[ThreadIdx.x].den*force.x;
	local[ThreadIdx.x].fy = f_press + MU/local[ThreadIdx.x].den*force.y;
	local[ThreadIdx.x].fz = f_press + MU/local[ThreadIdx.x].den*force.z + local[ThreadIdx.x].mass*GRAV;

	return local;
}


__device__ Force calc_l_visc(Particle *loc, Particle *bloc, p_type h) {
	int del = 0;
	Force force;
	force.x = force.y = force.z = 0;
	Particle me = loc[id];
	Particle you;
	p_type con;
	p_type h6 = pow(h,6);
	for(int i = 0; i < BlockDim.x; i++) {
		you = bloc[i];
		p_type r = sqrt((me.x - you.x)*(me.x - you.x) + (me.y - you.y)*(me.y - you.y) + (me.z - you.z)*(me.z - you.z));
		con = ((r-h)>>(sizeof(p_type)*8 -1)) * you.mass/you.den*45/(M_PI*h6);
		force.x += (you.vx-me.vx) * con;
		force.y += (you.vy-me.vy) * con;
		force.z += (you.vz-me.vz) * con;
	}
	return force;
}

__device__ p_type calc_l_pf(Particle *loc, Particle *bloc, p_type h) {
	int id = ThreadIdx.x;
	p_type f = 0;
	int del = 0;
	Particle me = loc[id];
	Particle you;
	p_type h6 = pow(h,6);
	for(int i = 0; i < BlockDim.x; i++) {
		you = bloc[i];
		p_type r = sqrt((me.x - you.x)*(me.x - you.x) + (me.y - you.y)*(me.y - you.y) + (me.z - you.z)*(me.z - you.z));
		f += ((r-h)>>(sizeof(p_type)*8 -1)) * me.mass * K_CON * (2*RHO + me.den + you.den) * 15/(M_PI*h6)*pow(h-r, 3);
	}
	return f;
}

// Calculates density of particle and stores in the struct
__device__ void calc_den(Particle *data, Particle *local, p_type h) {
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				z = BlockIdx.z + i;
				y = BlockIdx.y + j;
				x = BlockIdx.x + k;
				if (x<0 || x > GridDim.x - 1) {
					continue;
				}
				if (y<0 || y > GridDim.y - 1) {
					continue;
				}
				if (z<0 || z > GridDim.z - 1) {
					continue;
				}
				int id = z*BlockDim.z*BlockDim.y*BlockDim.x + y*BlockDim.y*BlockDim.x + x*BlockDim.x;
				__shared__ Particle shared[1024];
				shared[ThreadIdx.x] = data[id + ThreadIdx.x];
				__syncthreads();
				den += calc_l_den(local, shared, h);
			}
		}
	}
	local[ThreadIdx.x].den = den;
}

__device__ p_type calc_l_den(Particle *loc, Particle *bloc, p_type h) {
	p_type den = 0;
	int id = ThreadIdx.x;
	int del = 0;
	for(int i = 0; i < BlockDim.x; i++) {
		p_type rsq = (loc[id].x - bloc[i].x)*(loc[id].x - bloc[i].x) + (loc[id].y - bloc[i].y)*(loc[id].y - bloc[i].y) + (loc[id].z - bloc[i].z)*(loc[id].z - bloc[i].z);
		den += ((rsq-h*h)>>(sizeof(p_type)*8 -1)) * loc[id].mass * 315*(hsq-rsq)/(64*M_PI*pow(h,9));
	}
	return den;
}

void set_positions(float *pos, int N) {
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			pos[i*N + j] = j;
			pos[2*(i*N + j)] = i;
			pos[3*(i*N + j)] = (rand() % 50) / 10.0;
		}
	}
}
