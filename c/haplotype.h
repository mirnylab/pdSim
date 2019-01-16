#include <stdlib.h>
#define ASSERT(a,...) { if (!(a)) {printf(__VA_ARGS__); printf("\n"); exit(1);} }

#define INIT_ARRAY_SIZE	100000000
#define EXPANSION_RATE	4

static haplo_t *H, *H_end, *H_capacity;

void update_haplotype(cell_t *c, double ds) {
#if TREE
	if (H_end >= H_capacity) {
		unsigned new_length = EXPANSION_RATE*(H_capacity - H);
		printf("New haplotype array length: %d", new_length);
		haplo_t *new_H = (haplo_t *)realloc(H, new_length*sizeof(haplo_t));
		ASSERT(new_H != NULL, "Could not expand memory for haplotype array.");
		unsigned delta_H = new_H - H;
		H_end += delta_H;
		H += delta_H;
		H_capacity = &H[new_length];
		haplo_t *hp = H;
		while (hp->parent == NULL) hp++;
		for (; hp < H_end; hp++ )					  hp->parent += delta_H;
		for (cell_t *cp=Cells; cp < &Cells[n]; cp++ ) cp->h      += delta_H;
	}
	H_end->parent = c->h;
    H_end->ds = ds;
	c->h = H_end++;
#endif
}

void init_tree(double *fitness_array) {
	if (H != NULL) free(H);
	H = (haplo_t *)malloc(INIT_ARRAY_SIZE*sizeof(haplo_t));
	ASSERT(H != NULL, "Could not allocate memory for haplotype array.");
	H_end = H;
    H_capacity = &H[INIT_ARRAY_SIZE];
	for (int i = 0; i < n; i++) {
		H[i].parent = (haplo_t *)0; /*NULL;*/
		update_haplotype(Cells + i, fitness_array[i]);
	}
}

/* Process Phylogeny */
unsigned long get_haplotype(unsigned long i) { 
#if TREE
	return (i < n) ? (unsigned long)(Que[i]->h) : 0;
#endif
	return 0;
}

double get_fitness(unsigned long i) { return (i < n) ? 1./(Que[i]->bi) : -1; }

double collect_driver() {
	static unsigned long i = 0;
	while (i < H_end - H - 1) {
		double ds = H[i++].ds;
		if (ds > 0) return EPISTASIS == 1 ? -ds : ds/(1 - ds); 
	}
	return -1.;
}

