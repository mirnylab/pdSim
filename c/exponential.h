#pragma once
/* Text wrapping is not recommended for this document 
 * 
 * Exponential PRNG generator. Must call exponential_setup() once to initialize
 * the upstream uniform PRNG.
 *
 * exponential() -> Exponentially-distributed PRN with mean 1. 
 * */

#include "shared.h"

#define	__EXP_LAYERS__	252

/* The precomputed ziggurat lengths, denoted X_i in the main text (from create_layers.py) */
static double __exp_X__[__EXP_LAYERS__+1] = { 8.20662406753e-19, 7.39737323516e-19, 6.91333133779e-19, 6.5647358821e-19, 6.29125399598e-19, 6.06572241296e-19, 5.87352761037e-19, 5.70588505285e-19, 5.55709456916e-19, 5.42324389037e-19, 5.30152976965e-19, 5.18987392577e-19, 5.0866922618e-19, 4.99074929388e-19, 4.90106258944e-19, 4.81683790106e-19, 4.73742386536e-19, 4.66227958072e-19, 4.59095090178e-19, 4.52305277907e-19, 4.45825588164e-19, 4.39627631264e-19, 4.33686759671e-19, 4.27981436185e-19, 4.22492730271e-19, 4.17203912535e-19, 4.12100125225e-19, 4.07168112259e-19, 4.0239599631e-19, 3.97773093429e-19, 3.93289757853e-19, 3.88937251293e-19, 3.84707632187e-19, 3.80593661382e-19, 3.76588721385e-19, 3.7268674692e-19, 3.68882164922e-19, 3.65169842488e-19, 3.61545041533e-19, 3.58003379153e-19, 3.54540792845e-19, 3.51153509888e-19, 3.478380203e-19, 3.44591052889e-19, 3.41409553966e-19, 3.38290668387e-19, 3.35231722623e-19, 3.32230209587e-19, 3.29283775028e-19, 3.26390205282e-19, 3.23547416228e-19, 3.20753443311e-19, 3.18006432505e-19, 3.15304632118e-19, 3.12646385343e-19, 3.10030123469e-19, 3.07454359701e-19, 3.049176835e-19, 3.02418755411e-19, 2.99956302321e-19, 2.97529113107e-19, 2.95136034631e-19, 2.92775968057e-19, 2.90447865454e-19, 2.88150726664e-19, 2.85883596399e-19, 2.83645561563e-19, 2.81435748768e-19, 2.79253322026e-19, 2.77097480612e-19, 2.74967457073e-19, 2.72862515379e-19, 2.70781949192e-19, 2.68725080264e-19, 2.66691256932e-19, 2.64679852713e-19, 2.62690264997e-19, 2.60721913814e-19, 2.58774240685e-19, 2.56846707542e-19, 2.54938795718e-19, 2.53050004991e-19, 2.51179852691e-19, 2.49327872862e-19, 2.47493615466e-19, 2.45676645638e-19, 2.43876542983e-19, 2.42092900908e-19, 2.40325326001e-19, 2.38573437435e-19, 2.36836866406e-19, 2.35115255607e-19, 2.33408258722e-19, 2.31715539953e-19, 2.3003677357e-19, 2.28371643478e-19, 2.2671984282e-19, 2.2508107358e-19, 2.23455046227e-19, 2.21841479361e-19, 2.20240099382e-19, 2.18650640175e-19, 2.17072842808e-19, 2.15506455249e-19, 2.13951232087e-19, 2.12406934276e-19, 2.10873328882e-19, 2.09350188851e-19, 2.07837292773e-19, 2.06334424671e-19, 2.04841373792e-19, 2.03357934403e-19, 2.01883905608e-19, 2.00419091156e-19, 1.98963299272e-19, 1.97516342486e-19, 1.96078037473e-19, 1.94648204892e-19, 1.93226669243e-19, 1.9181325872e-19, 1.90407805074e-19, 1.89010143478e-19, 1.87620112397e-19, 1.86237553469e-19, 1.8486231138e-19, 1.83494233754e-19, 1.82133171034e-19, 1.80778976379e-19, 1.79431505561e-19, 1.78090616856e-19, 1.76756170954e-19, 1.75428030858e-19, 1.74106061794e-19, 1.7279013112e-19, 1.71480108238e-19, 1.7017586451e-19, 1.68877273172e-19, 1.67584209255e-19, 1.66296549505e-19, 1.65014172306e-19, 1.63736957602e-19, 1.62464786823e-19, 1.61197542813e-19, 1.59935109756e-19, 1.58677373107e-19, 1.57424219521e-19, 1.56175536784e-19, 1.54931213746e-19, 1.5369114025e-19, 1.52455207068e-19, 1.51223305837e-19, 1.49995328986e-19, 1.48771169674e-19, 1.47550721726e-19, 1.46333879563e-19, 1.4512053814e-19, 1.43910592874e-19, 1.42703939586e-19, 1.41500474425e-19, 1.40300093807e-19, 1.39102694344e-19, 1.37908172772e-19, 1.36716425886e-19, 1.35527350466e-19, 1.34340843201e-19, 1.3315680062e-19, 1.31975119012e-19, 1.3079569435e-19, 1.29618422208e-19, 1.28443197683e-19, 1.27269915307e-19, 1.26098468959e-19, 1.24928751776e-19, 1.23760656057e-19, 1.22594073168e-19, 1.21428893439e-19, 1.20265006056e-19, 1.19102298955e-19, 1.17940658704e-19, 1.16779970383e-19, 1.15620117456e-19, 1.14460981638e-19, 1.13302442758e-19, 1.12144378607e-19, 1.10986664787e-19, 1.0982917454e-19, 1.08671778581e-19, 1.07514344905e-19, 1.06356738599e-19, 1.05198821625e-19, 1.04040452605e-19, 1.02881486575e-19, 1.01721774741e-19, 1.00561164199e-19, 9.93994976483e-20, 9.82366130767e-20, 9.70723434263e-20, 9.59065162307e-20, 9.47389532242e-20, 9.35694699202e-20, 9.23978751546e-20, 9.12239705906e-20, 9.00475501809e-20, 8.88683995826e-20, 8.76862955198e-20, 8.65010050861e-20, 8.53122849831e-20, 8.41198806844e-20, 8.29235255165e-20, 8.1722939648e-20, 8.05178289728e-20, 7.93078838751e-20, 7.80927778595e-20, 7.68721660284e-20, 7.5645683384e-20, 7.44129429302e-20, 7.31735335451e-20, 7.19270175876e-20, 7.06729281977e-20, 6.94107662395e-20, 6.81399968293e-20, 6.68600453746e-20, 6.55702930402e-20, 6.42700715334e-20, 6.29586570809e-20, 6.16352634381e-20, 6.02990337322e-20, 5.89490308929e-20, 5.75842263599e-20, 5.62034866696e-20, 5.48055574135e-20, 5.3389043909e-20, 5.1952387718e-20, 5.04938378663e-20, 4.90114152226e-20, 4.75028679334e-20, 4.59656150013e-20, 4.4396673898e-20, 4.27925663021e-20, 4.11491932734e-20, 3.94616667626e-20, 3.77240771314e-20, 3.59291640862e-20, 3.40678366911e-20, 3.21284476416e-20, 3.00956469164e-20, 2.79484694556e-20, 2.56569130487e-20, 2.31752097568e-20, 2.04266952283e-20, 1.72617703302e-20, 1.32818892594e-20, 0.0};

void exponential_setup(void){
	mt_init();                                          /* Generates seed and fills uniform PRNG array */
}

static inline double _exp_overhang(uint_fast8_t j) {    /* Draws a PRN from overhang i */
    double *X_j = __exp_X__ + j;
	static double Y[__EXP_LAYERS__+1] = { 5.59520549511e-23, 1.18025099827e-22, 1.84444233867e-22, 2.54390304667e-22, 3.27376943115e-22, 4.03077321327e-22, 4.81254783195e-22, 5.61729148966e-22, 6.44358205404e-22, 7.29026623435e-22, 8.15638884563e-22, 9.04114536835e-22, 9.94384884864e-22, 1.0863906046e-21, 1.18007997755e-21, 1.27540755348e-21, 1.37233311764e-21, 1.47082087944e-21, 1.57083882574e-21, 1.67235819844e-21, 1.7753530675e-21, 1.87979997851e-21, 1.98567765878e-21, 2.09296677041e-21, 2.201649701e-21, 2.31171038523e-21, 2.42313415161e-21, 2.53590759014e-21, 2.65001843742e-21, 2.76545547637e-21, 2.88220844835e-21, 3.00026797575e-21, 3.11962549361e-21, 3.24027318888e-21, 3.36220394642e-21, 3.48541130074e-21, 3.60988939279e-21, 3.7356329311e-21, 3.86263715686e-21, 3.99089781236e-21, 4.12041111239e-21, 4.25117371845e-21, 4.38318271516e-21, 4.51643558895e-21, 4.65093020852e-21, 4.78666480711e-21, 4.92363796621e-21, 5.06184860075e-21, 5.20129594544e-21, 5.34197954236e-21, 5.48389922948e-21, 5.62705513018e-21, 5.77144764362e-21, 5.9170774359e-21, 6.06394543192e-21, 6.21205280795e-21, 6.36140098478e-21, 6.51199162141e-21, 6.66382660935e-21, 6.81690806729e-21, 6.97123833635e-21, 7.12681997563e-21, 7.28365575824e-21, 7.44174866764e-21, 7.60110189437e-21, 7.76171883308e-21, 7.92360307983e-21, 8.08675842978e-21, 8.25118887504e-21, 8.41689860281e-21, 8.58389199384e-21, 8.752173621e-21, 8.92174824817e-21, 9.0926208293e-21, 9.26479650768e-21, 9.43828061539e-21, 9.61307867302e-21, 9.78919638943e-21, 9.96663966183e-21, 1.01454145759e-20, 1.03255274063e-20, 1.05069846171e-20, 1.06897928622e-20, 1.08739589867e-20, 1.10594900275e-20, 1.12463932147e-20, 1.14346759725e-20, 1.16243459211e-20, 1.18154108781e-20, 1.20078788602e-20, 1.22017580851e-20, 1.23970569735e-20, 1.25937841516e-20, 1.27919484529e-20, 1.29915589212e-20, 1.31926248126e-20, 1.33951555991e-20, 1.35991609708e-20, 1.38046508394e-20, 1.40116353411e-20, 1.42201248406e-20, 1.44301299338e-20, 1.46416614524e-20, 1.48547304671e-20, 1.50693482921e-20, 1.5285526489e-20, 1.55032768718e-20, 1.57226115107e-20, 1.59435427376e-20, 1.61660831506e-20, 1.63902456195e-20, 1.6616043291e-20, 1.68434895946e-20, 1.70725982479e-20, 1.73033832633e-20, 1.75358589536e-20, 1.77700399393e-20, 1.80059411545e-20, 1.82435778548e-20, 1.84829656238e-20, 1.87241203814e-20, 1.89670583912e-20, 1.92117962687e-20, 1.94583509899e-20, 1.97067399002e-20, 1.99569807232e-20, 2.02090915706e-20, 2.04630909515e-20, 2.07189977831e-20, 2.09768314011e-20, 2.12366115708e-20, 2.14983584983e-20, 2.17620928428e-20, 2.20278357286e-20, 2.2295608758e-20, 2.2565434025e-20, 2.28373341287e-20, 2.31113321878e-20, 2.33874518561e-20, 2.36657173374e-20, 2.39461534023e-20, 2.42287854051e-20, 2.4513639301e-20, 2.48007416649e-20, 2.50901197103e-20, 2.53818013093e-20, 2.56758150136e-20, 2.59721900756e-20, 2.62709564716e-20, 2.65721449254e-20, 2.68757869323e-20, 2.71819147857e-20, 2.74905616033e-20, 2.78017613558e-20, 2.81155488957e-20, 2.84319599887e-20, 2.87510313451e-20, 2.90728006545e-20, 2.939730662e-20, 2.97245889962e-20, 3.00546886272e-20, 3.03876474879e-20, 3.07235087261e-20, 3.10623167078e-20, 3.14041170641e-20, 3.17489567409e-20, 3.20968840504e-20, 3.24479487265e-20, 3.28022019823e-20, 3.31596965706e-20, 3.35204868483e-20, 3.38846288435e-20, 3.42521803272e-20, 3.46232008885e-20, 3.4997752014e-20, 3.53758971719e-20, 3.57577019011e-20, 3.61432339058e-20, 3.65325631548e-20, 3.69257619879e-20, 3.73229052281e-20, 3.77240703013e-20, 3.81293373632e-20, 3.85387894342e-20, 3.89525125438e-20, 3.93705958834e-20, 3.97931319704e-20, 4.02202168223e-20, 4.06519501444e-20, 4.10884355286e-20, 4.15297806682e-20, 4.19760975869e-20, 4.24275028853e-20, 4.28841180055e-20, 4.3346069516e-20, 4.38134894182e-20, 4.42865154775e-20, 4.47652915804e-20, 4.52499681207e-20, 4.57407024181e-20, 4.62376591717e-20, 4.67410109528e-20, 4.72509387408e-20, 4.77676325071e-20, 4.82912918521e-20, 4.88221267023e-20, 4.93603580729e-20, 4.99062189052e-20, 5.04599549866e-20, 5.10218259653e-20, 5.15921064692e-20, 5.21710873452e-20, 5.2759077033e-20, 5.33564030933e-20, 5.39634139104e-20, 5.45804805963e-20, 5.52079991245e-20, 5.58463927299e-20, 5.64961146142e-20, 5.71576510093e-20, 5.7831524655e-20, 5.85182987638e-20, 5.92185815588e-20, 5.99330314883e-20, 6.06623632468e-20, 6.14073547584e-20, 6.21688553205e-20, 6.29477951501e-20, 6.37451966432e-20, 6.45621877375e-20, 6.54000178819e-20, 6.62600772633e-20, 6.71439201451e-20, 6.80532934473e-20, 6.89901720881e-20, 6.99568031586e-20, 7.09557617949e-20, 7.19900227889e-20, 7.30630537391e-20, 7.41789382663e-20, 7.53425421342e-20, 7.65597421711e-20, 7.78377498634e-20, 7.9185582674e-20, 8.06147755374e-20, 8.21405027698e-20, 8.37834459783e-20, 8.55731292497e-20, 8.75544596696e-20, 8.98023880577e-20, 9.24624714212e-20, 9.5919641345e-20, 1.08420217249e-19};	
	MT_FLUSH();
#ifdef SIMPLE_OVERHANGS
    double x = _FAST_PRNG_SAMPLE_X(X_j, RANDOM_INT63());   
                                            /* if y < f(x) return x, otherwise try again */
    return _FAST_PRNG_SAMPLE_Y(j, RANDOM_INT63()) <= exp(-x) ? x : _exp_overhang(j);    
#else
    int64_t U_x = RANDOM_INT63();               /* To sample a unit right-triangle: */
    int64_t U_distance = RANDOM_INT63() - U_x;  /* U_x <- min(U_1, U_2)             */
    if (U_distance < 0) {                       /* distance <- | U_1 - U_2 |        */
        U_distance = -U_distance;               /* U_y <- 1 - (U_x + distance)      */
        U_x -= U_distance;
    }
    static int64_t iE_max = 853965788476313646;                                     
    double x = _FAST_PRNG_SAMPLE_X(X_j, U_x);   
    if (U_distance >= iE_max) return x;     /* Early Exit: x < y - epsilon */ 
    return _FAST_PRNG_SAMPLE_Y(j, pow(2, 63) - (U_x + U_distance)) <= exp(-x) ? x : _exp_overhang(j); 
#endif
}


static uint_fast8_t _exp_sample_A(void) {
    /* Alias Sampling, see http://scorevoting.net/WarrenSmithPages/homepage/sampling.abs */
	static uint8_t map[256] = { 0, 0, 1, 235, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 250, 250, 250, 250, 250, 250, 250, 249, 249, 249, 249, 249, 249, 248, 248, 248, 248, 247, 247, 247, 247, 246, 246, 246, 245, 245, 244, 244, 243, 243, 242, 241, 241, 240, 239, 237, 3, 3, 4, 4, 6, 0, 0, 0, 0, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 2, 0, 0, 0 };
    static int64_t ipmf[256] = { 9223372036854775328, 1623796909450838420, 2664290944894293715, 7387971354164060121, 6515064486552739054, 8840508362680717952, 6099647593382935246, 7673130333659513959, 6220332867583438265, 5045979640552813853, 4075305837223955667, 3258413672162525563, 2560664887087762661, 1957224924672899759, 1429800935350577626, 964606309710808357, 551043923599587249, 180827629096890397, -152619738120023526, -454588624410291449, -729385126147774875, -980551509819446846, -1211029700667463872, -1423284293868547154, -1619396356369050292, -1801135830956212822, -1970018048575618008, -2127348289059705241, -2274257249303686299, -2411729520096655228, -2540626634159182525, -2661705860113406462, -2775635634532448735, -2883008316030465121, -2984350790383654722, -3080133339198118434, -3170777096303091107, -3256660348483804932, -3338123885075152741, -3415475560473282822, -3488994201966444710, -3558932970354470759, -3625522261068041096, -3688972217741992040, -3749474917563782729, -3807206277531056234, -3862327722496827274, -3914987649156779787, -3965322714631865323, -4013458973776912076, -4059512885612767084, -4103592206186241133, -4145796782586128173, -4186219260694363437, -4224945717447258894, -4262056226866285614, -4297625367836519694, -4331722680528537423, -4364413077437472623, -4395757214229418223, -4425811824915119504, -4454630025296932688, -4482261588141311280, -4508753193105271888, -4534148654077804689, -4558489126279970065, -4581813295192216657, -4604157549138257681, -4625556137145250418, -4646041313519109426, -4665643470413305970, -4684391259530326642, -4702311703971761747, -4719430301145086931, -4735771117539946355, -4751356876102103699, -4766209036859128403, -4780347871386013331, -4793792531638892019, -4806561113635122292, -4818670716409312756, -4830137496634465780, -4840976719260854452, -4851202804490332660, -4860829371376460084, -4869869278311657652, -4878334660640771092, -4886236965617427412, -4893586984900802772, -4900394884772702964, -4906670234238885493, -4912422031164489589, -4917658726580136309, -4922388247283532373, -4926618016851059029, -4930354975163335189, -4933605596540651285, -4936375906575303797, -4938671497741365845, -4940497543854575637, -4941858813449629493, -4942759682136114997, -4943204143989086773, -4943195822025527893, -4942737977813206357, -4941833520255033237, -4940485013586738773, -4938694684624359381, -4936464429291795925, -4933795818458825557, -4930690103114057941, -4927148218896868949, -4923170790008275925, -4918758132519202261, -4913910257091645845, -4908626871126550421, -4902907380349522964, -4896750889844289364, -4890156204540514772, -4883121829162554452, -4875645967641803284, -4867726521994894420, -4859361090668136340, -4850546966345097428, -4841281133215539220, -4831560263698486164, -4821380714613453652, -4810738522790066260, -4799629400105482131, -4788048727936313747, -4775991551010508883, -4763452570642098131, -4750426137329511059, -4736906242696389331, -4722886510751361491, -4708360188440098835, -4693320135461437394, -4677758813316075410, -4661668273553512594, -4645040145179234642, -4627865621182772242, -4610135444140937425, -4591839890849345681, -4572968755929937937, -4553511334358205905, -4533456402849118097, -4512792200036279121, -4491506405372581072, -4469586116675402576, -4447017826233108176, -4423787395382268560, -4399880027458432847, -4375280239014115151, -4349971829190464271, -4323937847117722127, -4297160557210950158, -4269621402214950094, -4241300963840749518, -4212178920821845518, -4182234004204468173, -4151443949668868493, -4119785446662289613, -4087234084103201932, -4053764292396157324, -4019349281473091724, -3983960974549676683, -3947569937258407435, -3910145301787369227, -3871654685619016074, -3832064104425399050, -3791337878631545353, -3749438533114317833, -3706326689447995081, -3661960950051848712, -3616297773528535240, -3569291340409179143, -3520893408440946503, -3471053156460654726, -3419717015797782918, -3366828488034800645, -3312327947826472069, -3256152429334011012, -3198235394669703364, -3138506482563184963, -3076891235255163586, -3013310801389731586, -2947681612411375617, -2879915029671665601, -2809916959107518656, -2737587429961872959, -2662820133571326270, -2585501917733374398, -2505512231579382333, -2422722515205206076, -2336995527534112187, -2248184604988688954, -2156132842510798521, -2060672187261006776, -1961622433929382455, -1858790108950092598, -1751967229002903349, -1640929916937143604, -1525436855617592627, -1405227557075244850, -1280020420662660017, -1149510549536587824, -1013367289578705710, -871231448632088621, -722712146453685035, -567383236774420522, -404779231966955560, -234390647591522471, -55658667960120229, 132030985907824093, 329355128892810847, 537061298001092449, 755977262693571427, 987022116608031845, 1231219266829421544, 1489711711346525930, 1763780090187560429, 2054864117341776240, 2364588157623792755, 2694791916990483702, 3047567482883492729, 3425304305830814717, 3830744187097279873, 4267048975685831301, 4737884547990035082, 5247525842198997007, 5800989391535354004, 6404202162993293978, 7064218894258529185, 7789505049452340392, 8590309807749443504, 7643763810684498323, 8891950541491447639, 5457384281016226081, 9083704440929275131, 7976211653914439517, 8178631350487107662, 2821287825726743868, 6322989683301723979, 4309503753387603546, 4685170734960182655, 8404845967535219911, 7330522972447586582, 1960945799077017972, 4742910674644930459, -751799822533465632, 7023456603741994979, 3843116882594690323, 3927231442413903597, -9223372036854775807, -9223372036854775807, -9223372036854775807 };
    uint_fast8_t j = Rand->sl & 0xff;           /* j <- I(0, 256) */
    return Rand++->sl >= ipmf[j] ? map[j] : j;
}

static inline double exponential(void) {
#ifndef INFER_TIMINGS
    static uint_fast8_t i_max = __EXP_LAYERS__;
#else
    static uint_fast8_t i_max = 2*__EXP_LAYERS__ - 256;
#endif
    MT_FLUSH();                                                 /* Refills array of Uniform PRNGs (when empty) */
    uint_fast8_t i = Rand->sl & 0xff;                           /* Float multiplication squashes these last 8 bits, so they can be used to sample i */
    if (i < i_max) return __exp_X__[i]*RANDOM_INT63();          /* Early Exit */
    Rand++;
    uint_fast8_t j = _exp_sample_A();                           /* from shared.h */ 
    static double X_0 = 7.56927469415;                          /* Beginning of tail */
    return j > 0 ? _exp_overhang(j) : X_0 + exponential();      /* sample from tail if j == 0; otherwise sample the overhang j */
}
