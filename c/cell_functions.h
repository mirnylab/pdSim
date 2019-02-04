/* Everything is much easier to handle as multiple files */
#include "debug.h"
#include "normal.h"
#include "exponential.h"

typedef struct haplo_t {        // The name for a haplotype in c is haplo-types, get it?
  struct haplo_t *parent;		// Every haplotype has a parent haplotype, thus defining the structure of the tree. 
  double ds;					// Every haplotype has a fitness that differs from its parent by ds. 
  double t;
} haplo_t;

typedef struct geno_t {			/* table entry: */
    struct geno_t *next; /* next entry in chain */
    __uint128_t genome; /* defined genome */
	double fitness;		/* fitness benefit due to drivers */
} geno_t;

typedef struct cell_t {
  double bi;			// Inverse birth rate, or inverse cell fitness
  double bt;			// Time until next birth
#if TREE
  haplo_t *h;
#endif
#if EXPLICIT_DRIVERS
  geno_t *genome;
#endif
} cell_t;

static cell_t *Cells, **Que;	// Global array of all cells, an array of pointers to every cell  
static double sd, sp, t, L;
static unsigned long n, nmin, nmax;

static inline double gamma_distribution() {
  double x = exponential();
  for (unsigned int i = 1; i <= SHAPE; i++) x += exponential();
  return x;
}

#define DISTRIBUTION(type) 	( type == 0 ? 1						: \
							( type == 1 ? exponential()			: \
							( type == 2 ? exp(SHAPE*normal())	: \
							( type == 3 ? gamma_distribution()	: \
							( type == 4 ? 2*uniform_double_PRN(): 1 )))))

#include "haplotype.h"
#include "genotype.h"

#define DRIVER_INTERACTION(dn) ( EPISTASIS == 1 ? (dn)*sd : (dn)*sd/(1+(dn)*sd) )

////////////////////////////////////////////////////////////////////////////////
// MUTATE.h                                 ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void inline update(cell_t *c, double ds) {
#if TREE
	update_haplotype(c, ds);
#endif
    c->bi *= EPISTASIS == 1 ? 1/(1+ds*c->bi) : 1-ds;
}

static void passenger_mutate(cell_t *c, double p) {
	if (p <= L) return;
	update(c, -sp*DISTRIBUTION(PASSENGER_DISTRIBUTION)); 
	passenger_mutate(c, p*uniform_double_PRN()); 
}

#define BIRTH(x) { x->bt = t + exponential()*x->bi; }
static unsigned long Ud_int;

static void mutate(cell_t *c) {
	MT_FLUSH();
	if (Rand++->l <= Ud_int) {
#if EXPLICIT_DRIVERS
		double old_fitness = c->genome->fitness;	
		c->genome = explicit_driver(c->genome);
		update(c, DRIVER_INTERACTION(c->genome->fitness - old_fitness));
#else
		update(c, DRIVER_INTERACTION(DISTRIBUTION(DRIVER_DISTRIBUTION)));
#endif  
}	
	passenger_mutate(c, uniform_double_PRN());
	BIRTH(c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Rest
/////////////////////////////////////////////////////////////////////////////////////////////////

/* See http://www.cs.caltech.edu/courses/cs191/paperscs191/JPhysChemA(2000-104)1876.pdf to understand Que 
The simulation can never have more than nmax cells. When there are fewer than nmax cells, the 'dead' cells are moved to the end of the que and assigned a birth time of INFINITY */
static cell_t **QueP1, **QueM1, sentinel_old, sentinel_young;		// Pointers and sentinels to check bounds of queues and update with maximum efficiency 

static int cell_compare(const void *a, const void *b) { return ( ((cell_t*)a)->bt > ((cell_t*)b)->bt  ? 1 : -1 ); }	// Are two genotypes identical?

void setup(unsigned long N_min, unsigned long N_max, double U_d, double U_p, double s_d, double s_p) {
    mt_init();
	nmin = N_min;
	nmax = N_max;
	Ud_int = (unsigned long)(U_d*pow(2, 64));
	sd = s_d;
	sp = s_p;
	L = exp(-U_p);
// SETUP CELLS & QUE
	Cells = (cell_t*)calloc(nmax, sizeof(cell_t));
	Que = (cell_t**)malloc(sizeof(cell_t**)*2*nmax);
	sentinel_old.bt = INFINITY;
	sentinel_young.bt = -1;
	Que[0] = &sentinel_young;
	for (cell_t **q=Que + nmax; q<Que + 2*nmax; q++) *q = &sentinel_old;
	QueM1 = Que;
	Que++;
	QueP1 = Que + 1;
}

#define SWAP_CELL_POINTERS(a,b) { cell_t *temp_cell = *(a); *(a) = *(b); *(b) = temp_cell; }
#define CHECK_HEAP_ORDERED() {if ((*child)->bt >= (*parent)->bt) return; }

void later(cell_t **parent) {				// re-orders que for a cell now dividing at a *later* point in time. 
							// These two functions are the rate limiting step in this algorithm; bit-wise operations and tail-recursion are used for speed. 
  cell_t **child = Que + ((parent-QueM1) << 1);		// location of child cell in the heap queue (x << 1 = x*2 in base 2) 
  if (child[0]->bt > child[-1]->bt) child--;    	// Each cell has two children; need youngest child   
  CHECK_HEAP_ORDERED();
  SWAP_CELL_POINTERS(child, parent); 				
  later(child);						// May be younger than grandchild...
}  

void sooner(cell_t **child) { 							// Move a cell up the birth queue if birth time has decreased 
  cell_t **parent = Que + ((child-QueP1) >> 1); 				// parent is the parent event of child in the queue (x >> 1 = x/2 in base 2) 
  CHECK_HEAP_ORDERED();
  SWAP_CELL_POINTERS(child, parent);
  sooner(parent);								// and now the parent becomes the child in the next round 
} 

double generation_time() {
    double Bi_sum = 0;
    for (cell_t **q=Que; q < Que+n; q++) Bi_sum += (*q)->bi;
    return Bi_sum/(2*n);
}

unsigned long simulation(unsigned long N_0, unsigned long t_max, unsigned long *Nt, double *Di, haplo_t **CTCs, double *fitness_array) {
  n = N_0; 
  t = 0;
  double t_next = 0;
  cell_t *c = Cells;
  for (double *fitness = fitness_array; fitness < fitness_array + N_0; fitness++, c++) { 
    c->bi = 1./ *fitness;
    BIRTH(c);
  }
#if TREE 
  init_tree(fitness_array);
#endif
#if EXPLICIT_DRIVERS
  init_genomes();
#endif
  for (; c<Cells+nmax; c++) c->bt = INFINITY;
  qsort(Cells, n, sizeof(cell_t), cell_compare);			/* Start cells out in order for Que */
  c=Cells;
  for (cell_t **q=Que; q < Que + nmax; q++, c++) *q = c;
/* Need to assign all elements in queue to a cell location since I'll use these pointers (i.e. pointers Q[i] : i > n) to find vacant cells spots when division occurs--I never create or destroy pointers to new locations, just swap */
// RUN SIMULATION
  for (int i=0; i<t_max; i++) {
    Nt[i] = n;
    t_next += generation_time();
#if CTCS
    CTCs[i] = (Que[rand_long(n)])->h;
#endif
    while (t<t_next) {
      t += exponential()*Di[n];												/* calculate time until next death, assuming this is the next event */
      if (Que[0]->bt<t) {													/* Birth */	
        t = Que[0]->bt;														/* update current time (we did not actually die!) */
        *Que[n] = *Que[0];													/* replicate cell and place at end of queue */
        mutate(Que[n]); 
        sooner(Que + n);													/* New cell at bottom might move up */
        if (++n >= nmax) {
          Nt[++i] = nmax; 
          return i; }														/* pop exceeded max */
        mutate(Que[0]);														/* mutate both daughters */
        later(Que);															/* New cell at bottom might move up */
        } else {
        cell_t **c = Que + rand_long(n);									/* randomly select cell to be killed & decrease pop */
        if (--n <= nmin) {
          Nt[++i] = nmin; 
          return i; }														/* Extinction */
        SWAP_CELL_POINTERS(c, Que + n);										/* destroy cell by replacing with cell at end of queue--make sure end of queue points to the dead cell because birth events use this pointer to find a vacant spot in Cells. */
        Que[n]->bt = INFINITY;												/* dead cells don't divide */
        sooner(c);															/* cell could've been younger than its parent, if so we should move it up. */
        later(c);															/* else move cell at end of queue down */
  } } } 
  Nt[t_max] = n;
  return t_max;
} 

void end() { 
	free(Cells);	Cells = NULL; 
	Que--;			// Que was originally incremented for a sentinel	
	free(Que);		Que = NULL;	
	free(H);		H = NULL;
	deallocate_genome_hashtable();
}

