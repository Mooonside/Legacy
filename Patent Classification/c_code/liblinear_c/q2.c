#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>  
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void void_print_null(const char *s) {}

static char *line = NULL;
static int max_line_len;

struct feature_node *x_space;
struct parameter param;
struct problem prob;
int flag_cross_validation;
int flag_find_C;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;
//predict
struct feature_node *x;
int max_nr_attr = 64;
static int (*info)(const char *fmt,...) = &printf;
//q3
int *random_choice_0,*random_choice_1,*random_choice_2,*random_choice_3; 
int data_dist[2] = {0};
struct model ***model_list;
int split_positive;
int split_negative;


bool check_label(char* pointer){
	if(pointer == NULL)
		return false;
	else if(pointer[0] >= 'A' && pointer[0] <= 'Z')
		return true;
	else
		return false;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		//printf("reading in line:%s\n", line);
		char *p = strtok(line," "); // label
        do{
            p = strtok(NULL," ");
        }while(check_label(p));

		// features
		//printf("features:");
		while(1)
		{
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			//printf(" %s ",p);
			elements++;
			p = strtok(NULL," \t");
		}
		//printf("\n");
		////printf("%d has %d features \n",prob.l,int(elements));
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];

		label = strtok(line," ");
		prob.y[i] = label[0]-'A' + 0;

		if(prob.y[i] == 0){
			data_dist[0] += 1;				
		}else{
			data_dist[1] += 1;
		}
		//printf("%d sample has label:%f \n",i,prob.y[i]);
		if(label == NULL) // empty line
			exit_input_error(i+1);
		

        do{
            label = strtok(NULL," ");
        }while(check_label(label));	
		
		size_t temp = j;
		j++;
		////////////////////////////////////////////////////////////////////////////
		//!!!!!!HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/
		///////////////////////////////////////////////////////////////////////////
		//prob.y[i] = strtod(label,&endptr);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			//printf("%d sample's %d : %f\n",i,x_space[j].index,x_space[j].value );

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		idx= strtok(label,":");
		val = strtok(NULL,"\0");
		x_space[temp].index = (int) strtol(idx,&endptr,10);
		x_space[temp].value = strtod(val,&endptr);
		//printf("0 feature: %d : %d\n",x_space[temp].index,x_space[temp].value);

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	// for (int k=0;k<elements+prob.l;k++){
	// 	printf("%d %f\n",x_space[k].index,x_space[k].value);
	// }

	fclose(fp);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.init_sol = NULL;
	flag_cross_validation = 0;
	flag_C_specified = 0;
	flag_solver_specified = 0;
	flag_find_C = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				flag_solver_specified = 1;
				break;

			case 'c':
				param.C = atof(argv[i]);
				flag_C_specified = 1;
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				printf("doing %d -flod validation\n",nr_fold);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &void_print_null;
				i--;
				break;

			case 'C':
				flag_find_C = 1;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	// default solver for parameter selection is L2R_L2LOSS_SVC
	if(flag_find_C)
	{
		if(!flag_cross_validation)
			nr_fold = 5;
		if(!flag_solver_specified)
		{
			fprintf(stderr, "Solver not specified. Using -s 2\n");
			param.solver_type = L2R_L2LOSS_SVC;
		}
		else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC)
		{
			fprintf(stderr, "Warm-start parameter search only available for -s 0 and -s 2\n");
			exit_with_help();
		}
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}


void do_predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int n;
	int nr_feature=get_nr_feature(model_list[0][0]);
	if(model_list[0][0]->bias==0)
		n=nr_feature+1;
	else
		n=nr_feature;

	max_line_len = 10000;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		//printf("read test line:%s\n",line);
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," ");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = label[0] - 'A' + 0;
		
		if(target_label == 0)
			target_label = 1;
		else
			target_label = -1;

        do{
            label = strtok(NULL," ");
        }while(check_label(label));

		i = 1;
		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}
			
			idx = strtok(NULL,":");
			val = strtok(NULL," ");
			// printf("idx :%s\n",idx );
			// printf("val :%s\n",val );

			//printf("%s : %s\n",idx,val);
			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);

			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);


			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}

		idx= strtok(label,":");
		val = strtok(NULL,"\0");
		x[0].index = (int) strtol(idx,&endptr,10);
		x[0].value = strtod(val,&endptr);

		if(model_list[0][0]->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_list[0][0]->bias;
			i++;
		}
		x[i].index = -1;



		// for (int it=0;it<i;it++){
		// 	printf("%d : %f ,",x[it].index,x[it].value);
		// }
		// printf("\n");
		/////////////////////////////////////////////////////////////
		/////////////////have to change here!////////////////////////
		/////////////////////////////////////////////////////////////


		//predict_label = predict(model_list[0][0],x);
		double * nonsense;
		nonsense = Malloc(double,1);

		double max = -999999;
		for(int it = 0;it<split_positive;it++){
			double min = 999999;
			for(int it1 = 0;it1<split_negative;it1++){
				double temp =  predict_values(model_list[it][it1],x,nonsense);
				if(temp < min){
					min = temp;
				}
			}
			if(min > max){
				max = min;
			}
		}


		//printf("%f\n",nonsense);

		if(max > 0){
			predict_label = 1;
		}else{
			predict_label = -1;
		}

		fprintf(output,"%g\n",predict_label);



		if(predict_label == target_label)
			++correct;

		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}

	info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
}

void do_find_parameter_C()
{
	double start_C, best_C, best_rate;
	double max_C = 1024;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;
	printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
	find_parameter_C(&prob, &param, nr_fold, start_C, max_C, &best_C, &best_rate);
	printf("Best C = %g  CV accuracy = %g%%\n", best_C, 100.0*best_rate);
}


void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			  );
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}


void show_data(struct feature_node* x){
	int it = 0;
	while(true){
		if(x[it].index == -1){
			printf("\n");
			return;
		}else{
			printf("%d : %f ,",x[it].index,x[it].value);
			it += 1;
		}
	}
	return;
}

void shuffle(int *a, int n)  
{  
	int index, tmp, i;
	for (i = 0; i <n; i++)  
	{  
		a[i] = i;
	}	

	srand(time(NULL));  
	for (i = 0; i <n; i++)  
	{  
		index = rand() % (n - i) + i;  
		if (index != i)  
		{  
			tmp = a[i];  
			a[i] = a[index];  
			a[index] = tmp;  
		}  
	}	  
}


void mini_maker(struct problem &mini_prob,struct feature_node** x1,
						int p1,int size1,struct feature_node** x2,int p2,int size2){
	for (int it = 0;it<size1;it++){
		mini_prob.y[it] = 1;
		mini_prob.x[it] = x1[p1+it];
	}
	for(int it = 0;it<size2;it++){
		mini_prob.y[it+size1] = -1;
		mini_prob.x[it+size1] = x2[p2+it];
	}
	
}

int main(int argc, char **argv)
{
	clock_t tic,tiic,tiiic; 

	char input_file_name[1024];
	char test_file_name[1024];

	parse_command_line(argc, argv, input_file_name, test_file_name);
	printf("Train data:%s\n",input_file_name);
	printf("Test data:%s\n",test_file_name);

	read_problem(input_file_name);
	
	// now do data sepratation
	printf("%d\n",prob.l); 
	printf("data distribution: %d,%d\n",data_dist[0],data_dist[1]);


	struct feature_node **x_dist[2];
	for (int it=0;it<2;it++){
		x_dist[it] = Malloc(struct feature_node *,data_dist[it]);
	}

	int temp_it[2] = {0};
	for (int it = 0;it<prob.l;it++){
		if(prob.y[it] == 0){
			x_dist[0][temp_it[0]] = prob.x[it];				
			temp_it[0] ++;
		}else{
			x_dist[1][temp_it[1]] = prob.x[it];
			temp_it[1] ++;
		}
	}

	//display part
	// printf("class A :\n");
	// for (int it=0;it<5;it++){
	// 	printf("############################################\n");
	// 	show_data(x_dist[1][it]);
	// 	printf("############################################\n");
	// }

	random_choice_0 = Malloc(int,data_dist[0]);
	random_choice_1 = Malloc(int,data_dist[1]);
	shuffle(random_choice_0,data_dist[0]);
	shuffle(random_choice_1,data_dist[1]);

	// for (int it=0;it<5;it++){
	// 	printf("%d,",random_choice_0[it]);
	// }
	// printf("\n");

	int split[2] = {5,15};
	split_positive = split[0];
	split_negative = 0;
	for (int it=1;it<2;it++){
		split_negative += split[it];
	}
	printf("##########################################\n");
	printf("we got %d models to train!\n",split_positive*split_negative);
	printf("##########################################\n");

	int size[2] = {0};
	for (int it=0;it<2;it++){
		size[it] = int(data_dist[it]/split[it]);
	}

	printf("##########################################\n");
	printf("mini problem size:\n");
	for(int it=0;it<2;it++){
		printf("%d ",size[it]);
	}
	printf("\n");
	printf("##########################################\n");

	tic = clock();

	struct problem mini_prob;
	model_list = Malloc(struct model**,split_positive);
	for (int it=0;it < split_positive;it++){
		model_list[it] = Malloc(struct model*,split_negative);
	}

	struct model* mini_model;

	for (int it = 0;it<split[0];it++){
		mini_prob.n = prob.n;
		mini_prob.bias = prob.bias;
		
		for(int it1 = 0;it1<split[1];it1++){
			mini_prob.l = size[0] + size[1];
			mini_prob.y = Malloc(double,mini_prob.l);
			mini_prob.x = Malloc(struct feature_node*,mini_prob.l);
			mini_maker(mini_prob,x_dist[0],it*size[0],size[0],
				x_dist[1],it1*size[1],size[1]);
			mini_model = train(&mini_prob, &param);
			printf("model %d,%d is trained\n",it,it1);
			model_list[it][it1] = mini_model;
			free(mini_prob.y);
			free(mini_prob.x);
		}
	}

	tiic = clock();
	printf("successfully trained\n");

	char output_file_name[] = "ultra_prediction"; 
	
	FILE *test_input,*pred_output; 
	test_input = fopen(test_file_name,"r");
	if(test_input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",test_file_name);
		exit(1);
	}

	pred_output = fopen(output_file_name,"w");
	if(pred_output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",output_file_name);
		exit(1);
	}	

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	do_predict(test_input, pred_output);


	tiiic = clock();
	double train_time = ((double)(tiic-tic)/CLOCKS_PER_SEC);
	double pred_time = ((double)(tiiic-tiic)/CLOCKS_PER_SEC);

	printf("train time : %f s\n",train_time);
	printf("pred time : %f s\n",pred_time);
}