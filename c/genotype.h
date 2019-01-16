/* Explicitly model driver mutations occurring at specific sites in the cancer genome. 
 * This is mostly intended for simulations where specific driver mutations interact epistatically. 
 * The key function in this library is:
 *
 * double explicit_driver(__uint128_t *previous_extant_genome)
 *
 * , which will generate a new driver mutation from the previous existing genome. Genomes are defined
 * by a 128 bit integer, where each bit is 0 if the loci is un-mutated and 1, if the loci is mutated.
 * There are LOCI number of mutable sites in the genome and U_d remains the *genome-wide* mutation rate. 
 * The fitness of a new genome is calculated by driver_fitness(__uint128_t new_genome), which can have
 * various forms defined by the value of EXPLICIT_DRIVERS and SHAPE. 
 *
 * Because the fitness of various genomes are generated using random variables and because the set of 
 * possible generable genomes in this library is 2^128 >> system memory, this algorithm memoizes previously
 * randomly-generated genomes, to reuse if they are encountered again during evolution. This requires the
 * creation of a hashtable that must be created and destroyed using:
 *
 * void init_genomes()
 * void deallocate_genome_hashtable()
 *
 * This also required creation of two functions for explict_driver: 
 *
 * double genome_fitness_lookup(__uint128_t *genome)
 * void record_new_genome(__uint128_t genome, double fitness)
 *
 * , as well as an internal hash function and internal linked-list structure for the hash. 
 *
 * */

#define HASHBITS	12
#define HASHSIZE	pow(2, HASHBITS)
static struct geno_t **hashtable;	/* pointer table */

double driver_fitness(__uint128_t new_genome) {
	int n_d = __builtin_popcount(new_genome);
#if   EXPLICIT_DRIVERS == 1						// simple linear model of fitness
	return n_d;
#elif EXPLICIT_DRIVERS == 2						// Mt. Fuji model of fitness
	return n_d + SHAPE*normal();
#elif EXPLICIT_DRIVERS == 3						// two-hit model
	return MAX(0, n_d - 1);					
#elif EXPLICIT_DRIVERS == 4						// Other two-hit model
	return n_d == 1 ? 0 : n_d;
#endif
}

void deallocate_genome_hashtable() {
#if EXPLICIT_DRIVERS
	for (unsigned long i = 0; i < HASHSIZE; i++) {
		struct geno_t *gp = hashtable[i];
		while (gp != NULL) {
			struct geno_t *last_gp = gp;
			gp = last_gp->next;
			free(last_gp);
		}
	}
	free(hashtable);
	hashtable = NULL;				// To prevent over-deallocating 
#endif
}

#define  MERSENE_PRIME	2305843009213693951

/* look for genome in hashtable, create if needed, and return pointer to geno_t */
geno_t *find_or_add_new_genome(__uint128_t genome) {
    struct geno_t *gp;
				/* hash: form hash value for genome g, using Kunth's Multiplicative Hashing */
	unsigned long hashvalue = (genome*MERSENE_PRIME) >> (sizeof(__uint128_t)*8 - HASHBITS);
	assert(hashvalue < HASHSIZE);
	for (gp = hashtable[hashvalue]; gp != NULL; gp = gp->next)
        if (gp->genome == genome) return gp;	/* found */
	/* else, not found */
	struct geno_t *new_geno_t = (struct geno_t *)malloc(sizeof(struct geno_t));
	ASSERT(new_geno_t != NULL, "Could not allocate memory for new genome.");
	new_geno_t->genome = genome;
	new_geno_t->fitness = driver_fitness(genome);
	/* insert genome into first column of hash table */
	new_geno_t->next = hashtable[hashvalue];
	hashtable[hashvalue] = new_geno_t;
	return new_geno_t;	
}

/* Initialize hashtable and asign INIT_GENOME to each founding cell */
#define INIT_GENOME		0x0000000000000000 
void init_genomes() {
#if EXPLICIT_DRIVERS	
	ASSERT(LOCI <= sizeof(__uint128_t)*8, "Loci must be smaller than genome length");

	if (hashtable != NULL) deallocate_genome_hashtable();	// Sometimes a new hashtable is initialized before the old one is closed.
	hashtable = (struct geno_t**)calloc(HASHSIZE, sizeof(struct geno_t*));	// All genomes must initially be NULL.
	geno_t *init_genome = find_or_add_new_genome(INIT_GENOME);
	for (cell_t *cp = Cells; cp < Cells + n; cp++) 
		cp->genome = init_genome;
#endif
}

/* Find the fitness of an explicit driver mutation arising from previous_extant_genome */
geno_t *explicit_driver(geno_t *old_genotype) {
	unsigned long mutant_loci = rand_long(LOCI);
	if ( ( old_genotype->genome >> mutant_loci) & 1) return old_genotype;			// Mutated an already mutated loci
	return find_or_add_new_genome( old_genotype->genome | (1u << mutant_loci));
}

