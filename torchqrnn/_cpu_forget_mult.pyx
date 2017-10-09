# cython: infer_types=True
#

cdef void recurrent_forget_mult(float *dst, const float* F, const float* X,
        int nN, int nB, int nO) nogil:
    cdef int ts, b, h, prev_step, this_step
    # ts is timestep index, b is batch index, h is hidden index
    cdef int i = 0
    for ts in range(1, nN+1):
        # To move timesteps, we step HIDDEN * BATCH
        # To move batches, we move HIDDEN
        # To move neurons, we move +- 1
        prev_step = (ts-1) * nO * nB
        this_step = (ts-0) * nO * nB
        for b in range(nB):
            prev_step += b * nO
            this_step += b * nO
            for h in range(nO):
                dst[this_step]  = (F[i] * X[i]) + ((1 - F[i]) * dst[prev_step])
                i += 1
                this_step += 1
                prev_step += 1


cdef void bwd_recurrent_forget_mult(float *dF, float *dX, float *dHinit,
        const float *H, const float *F, const float *X, const float *dH,
        int nN, int nB, int nO) nogil:
    cdef double running_f = 0
    cdef int ts, b, h, prev_step, this_step
    cdef int i = 0
    for ts in range(nN, 1, -1):
        # ts is timestep index, b is batch index, h is hidden index
        # To move timesteps, we step HIDDEN * BATCH
        # To move batches, we move HIDDEN
        # To move neurons, we move +- 1
        prev_step = (ts-1) * nO * nB
        this_step = (ts-0) * nO * nB
        for b in range(nB):
            prev_step += b * nO
            this_step += b * nO
            for h in range(nO):
                running_f += dH[this_step]
                # Gradient of X
                dX[i] = F[i] * running_f
                # Gradient of F
                dF[i] = (X[i] - H[prev_step]) * running_f
                # Likely more numerically stable than (1 - F[i]) * running_f
                running_f = running_f - F[i] * running_f
                dHinit[i] = running_f
                i += 1
