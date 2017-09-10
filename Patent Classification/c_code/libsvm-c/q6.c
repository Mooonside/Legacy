#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define TRAIN_PTHREAD 5
#define PRED_PTHREAD 8

struct svm_parameter param;     // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;
static char *line = NULL;
static int max_line_len;
static int (*info)(const char *fmt,...) = &printf;
int max_nr_attr = 64;

//divide data
int *random_choice_0,*random_choice_1; 
struct svm_model ***model_list;
int split_positive;
int split_negative;
struct svm_node **x_dist[2];
int test_num = 0;
int split[2] = {5,15};
int size[2] = {0};
//q5
pthread_t *thread_pool;
int data_dist[2] = {0};
struct svm_node **test_x;
double *test_y;
double *pred_y;

void print_null(const char *s) {}

bool check_label(char* pointer){
   if(pointer == NULL)
      return false;
   else if(pointer[0] >= 'A' && pointer[0] <= 'Z')
      return true;
   else
      return false;
}


void exit_with_help()
{
    printf(
    "Usage: svm-train [options] training_set_file [model_file]\n"
    "options:\n"
    "-s svm_type : set type of SVM (default 0)\n"
    "   0 -- C-SVC      (multi-class classification)\n"
    "   1 -- nu-SVC     (multi-class classification)\n"
    "   2 -- one-class SVM\n"
    "   3 -- epsilon-SVR    (regression)\n"
    "   4 -- nu-SVR     (regression)\n"
    "-t kernel_type : set type of kernel function (default 2)\n"
    "   0 -- linear: u'*v\n"
    "   1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
    "   2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
    "   3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
    "   4 -- precomputed kernel (kernel values in training_set_file)\n"
    "-d degree : set degree in kernel function (default 3)\n"
    "-g gamma : set gamma in kernel function (default 1/num_features)\n"
    "-r coef0 : set coef0 in kernel function (default 0)\n"
    "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
    "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
    "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
    "-m cachesize : set cache memory size in MB (default 100)\n"
    "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
    "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
    "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
    "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
    "-v n: n-fold cross validation mode\n"
    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

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


void mini_maker(struct svm_problem &mini_prob,struct svm_node** x1,
                  int p1,int size1,struct svm_node** x2,int p2,int size2)
{
   for (int it = 0;it<size1;it++){
      mini_prob.y[it] = 1;
      mini_prob.x[it] = x1[p1+it];
   }
   for(int it = 0;it<size2;it++){
      mini_prob.y[it+size1] = -1;
      mini_prob.x[it+size1] = x2[p2+it];
   }
   
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

void read_test(FILE *input)
{
    int svm_type=svm_get_svm_type(model_list[0][0]);
    int nr_class=svm_get_nr_class(model_list[0][0]);
    double *prob_estimates=NULL;
    int j;

    max_line_len = 10000;
    line = (char *)malloc(max_line_len*sizeof(char));

    while(readline(input)!=NULL){
        test_num ++;
    }
    //printf("test_num:%d\n",test_num);
    rewind(input);

    test_y = Malloc(double,test_num);
    test_x = Malloc(struct svm_node*,test_num);
    
    int i = 0;
    while(readline(input) != NULL)
    {
        test_x[i] = Malloc(svm_node,max_nr_attr);
        int j = 0;
        double target_label, predict_label;
        char *idx, *val, *label, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);

        target_label = label[0] - 'A' + 0;


        if(target_label == 0){
            target_label = 1;
            test_y[i] = 1;
        }
        else{
            target_label = -1;
            test_y[i] = -1;
        }

        do{
            label = strtok(NULL," ");
        }while(check_label(label));

        j = 1;

        while(1)
        {
            if(j>=max_nr_attr-1)    // need one more for index = -1
            {
                max_nr_attr *= 2;
                test_x[i] = (struct svm_node *) realloc(test_x[i],max_nr_attr*sizeof(struct svm_node));
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;
            errno = 0;
            test_x[i][j].index = (int) strtol(idx,&endptr,10);

            if(endptr == idx || errno != 0 || *endptr != '\0' || test_x[i][j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = test_x[i][j].index;

            errno = 0;
            test_x[i][j].value = strtod(val,&endptr);

            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        idx= strtok(label,":");
        val = strtok(NULL,"\0");

        test_x[i][0].index = (int) strtol(idx,&endptr,10);
        test_x[i][0].value = strtod(val,&endptr);

        test_x[i][j].index = -1;
        
        ++ i;
    }
}

void show_data(struct svm_node* x){
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

void pred(){
    double temp;
   for(int i=0;i<test_num;i++){
        svm_predict_values(model,test_x[i],&temp);
        pred_y[i] = temp;
   }
}

void *thread_train(void *thread_param){
   int step = split_positive / TRAIN_PTHREAD;
   int start = *(int *) thread_param;
   int upper = step*(start+1);
   if(start == TRAIN_PTHREAD -1){
      upper = split_positive;
   }

   struct svm_problem mini_prob;
   struct svm_model* mini_model;

   for (int it = step*start;it < upper;it++){

      for(int it1 = 0;it1<split_negative;it1++){
         mini_prob.l = size[0] + size[1];
         mini_prob.y = Malloc(double,mini_prob.l);
         mini_prob.x = Malloc(struct svm_node*,mini_prob.l);
         mini_maker(mini_prob,x_dist[0],it*size[0],size[0],
            x_dist[1],it1*size[1],size[1]);
         mini_model = svm_train(&mini_prob, &param);
         printf("%d %d model is trained!\n",it,it1);
         //printf("model %d,%d is trained\n",it,it1);
         model_list[it][it1] = mini_model;
         free(mini_prob.y);
         free(mini_prob.x);
      }
   }
   pthread_exit(NULL); 
}

void *thread_pred(void *thread_param){
   int start = *(int *)thread_param;
   int step = test_num / PRED_PTHREAD;
   int upper = step * (start + 1);
   if(start == PRED_PTHREAD-1){
      upper = test_num;
   } 

   for(int i=start*step;i<upper;i++){
      double temp;
      double max = -999999;
      for(int it = 0;it<split_positive;it++){
         double min = 999999;
         for(int it1 = 0;it1<split_negative;it1++){
            svm_predict_values(model_list[it][it1],test_x[i],&temp);
            if(temp < min){
               min = temp;
            }
            if(it1!= 0 && it1 % 5000 == 0){
                printf("5000 predition from model %d %d\n",it,it1);
            }
        }

         if(min > max){
            max = min;
         }
      }
      pred_y[i] = max;
   }
   pthread_exit(NULL); 
}

int main(int argc, char **argv)
{
    struct timespec tic,tiic,tiiic;

    char input_file_name[1024];
    char test_file_name[1024];
    const char *error_msg;

    parse_command_line(argc, argv, input_file_name, test_file_name);
    printf("Train data:%s\n",input_file_name);
    printf("Test data:%s\n",test_file_name);

    read_problem(input_file_name);
    //show_data(prob.x[1]);

    printf("%d\n",prob.l); 
    printf("data distribution: %d,%d\n",data_dist[0],data_dist[1]);

    for (int it=0;it<2;it++){
        x_dist[it] = Malloc(struct svm_node *,data_dist[it]);
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

    random_choice_0 = Malloc(int,data_dist[0]);
    random_choice_1 = Malloc(int,data_dist[1]);
    shuffle(random_choice_0,data_dist[0]);
    shuffle(random_choice_1,data_dist[1]);

    split_positive = split[0];
    split_negative = 0;
    for (int it=1;it<2;it++){
        split_negative += split[it];
    }
    printf("##########################################\n");
    printf("we got %d models to train!\n",split_positive*split_negative);
    printf("##########################################\n");

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

    clock_gettime(CLOCK_MONOTONIC,&tic);



    model_list = Malloc(struct svm_model**,split_positive);
    for (int it=0;it < split_positive;it++){
        model_list[it] = Malloc(struct svm_model*,split_negative);
    }

    printf("##########################################\n");
    printf("Training! Using %d threads \n",TRAIN_PTHREAD);
    thread_pool= Malloc(pthread_t,TRAIN_PTHREAD);
    for(int it = 0;it<TRAIN_PTHREAD;it++){
        int *thread_param = new int(it);
        pthread_create(&thread_pool[it],NULL,&thread_train,thread_param);
    }

    for(int it=0;it<TRAIN_PTHREAD;it++){
        pthread_join(thread_pool[it],NULL);
    }
    free(thread_pool);
    
    clock_gettime(CLOCK_MONOTONIC,&tiic);
    printf("successfully trained\n");
    printf("##########################################\n");

    char output_file_name[] = "svm_pred_random";   
    
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

    read_test(test_input);
    printf("% d test data read in!\n",test_num);
    printf("##########################################\n");
    printf("Predicting! Using %d threads \n",PRED_PTHREAD);
    pred_y = Malloc(double,test_num);
    //predition!
    thread_pool= Malloc(pthread_t,PRED_PTHREAD);
    for(int it = 0;it<PRED_PTHREAD;it++){
        int *thread_param = new int(it);
        pthread_create(&thread_pool[it],NULL,&thread_pred,thread_param);
    }

    for(int it=0;it<PRED_PTHREAD;it++){
        pthread_join(thread_pool[it],NULL);
    }
    free(thread_pool);
    printf("Successfully predicted!\n");
    printf("##########################################\n");

    int correct = 0;
    double temp;
    for(int i=0;i<test_num;i++){
        fprintf(pred_output,"%g\n",pred_y[i]);
        //printf("pred value: %f \n",pred_y[i]);
        if(pred_y[i] >= 0){
            temp = 1;
        }else{
            temp = -1;
        }
        if(temp == test_y[i]){
            correct++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&tiiic);

    info("Accuracy = %g%% (%d/%d)\n",(double) correct/test_num*100,correct,test_num);

    double train_time = tiic.tv_sec - tic.tv_sec;
    train_time += (tiic.tv_nsec - tic.tv_nsec)/1000000000.0;
    
    double pred_time = tiiic.tv_sec - tiic.tv_sec;
    pred_time += (tiiic.tv_nsec - tiic.tv_nsec)/1000000000.0;

    printf("train time : %f s\n",train_time);
    printf("pred time : %f s\n",pred_time);
    return 0;
}

void do_cross_validation()
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double,prob.l);

    svm_cross_validation(&prob,&param,nr_fold,target);
    if(param.svm_type == EPSILON_SVR ||
       param.svm_type == NU_SVR)
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

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i;
    void (*print_func)(const char*) = NULL; // default printing to stdout

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;    // 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    cross_validation = 0;

    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {
            case 's':
                param.svm_type = atoi(argv[i]);
                break;
            case 't':
                param.kernel_type = atoi(argv[i]);
                break;
            case 'd':
                param.degree = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'r':
                param.coef0 = atof(argv[i]);
                break;
            case 'n':
                param.nu = atof(argv[i]);
                break;
            case 'm':
                param.cache_size = atof(argv[i]);
                break;
            case 'c':
                param.C = atof(argv[i]);
                break;
            case 'e':
                param.eps = atof(argv[i]);
                break;
            case 'p':
                param.p = atof(argv[i]);
                break;
            case 'h':
                param.shrinking = atoi(argv[i]);
                break;
            case 'b':
                param.probability = atoi(argv[i]);
                break;
            case 'q':
                print_func = &print_null;
                i--;
                break;
            case 'v':
                cross_validation = 1;
                nr_fold = atoi(argv[i]);
                if(nr_fold < 2)
                {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;
            case 'w':
                ++param.nr_weight;
                param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
                param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
                param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
                param.weight[param.nr_weight-1] = atof(argv[i]);
                break;
            default:
                fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
        }
    }

    svm_set_print_string_function(print_func);

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
}

// read in a problem (in svmlight format)

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

    max_line_len = 10000;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        //printf("%s\n",line);
        char *p = strtok(line," \t"); // label
        // while(check_label(p)){
        //     p = strtok(NULL," ");
        // }
        do{
            p = strtok(NULL," ");
        }while(check_label(p));
        // features
        while(1)
        {
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
            p = strtok(NULL," \t");
        }
        ++elements;
        ++prob.l;
    }

    //printf("probem size:%d",prob.l);
    rewind(fp);

    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct svm_node *,prob.l);
    x_space = Malloc(struct svm_node,elements);

    max_index = 0;
    j=0;
    for(i=0;i<prob.l;i++)
    {
        //printf("reading %d line\n",i);
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line," ");
        prob.y[i] = label[0] - 'A' + 0;

        if(prob.y[i] == 0){
            data_dist[0] += 1;                
        }else{
            data_dist[1] += 1;
        }

        if(label == NULL) // empty line
            exit_input_error(i+1);

        do{
            label = strtok(NULL," ");
        }while(check_label(label));      

        size_t temp = j;
        j++;

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
            
            //printf("%d:%f\n",x_space[j].index,x_space[j].value);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        idx= strtok(label,":");
        val = strtok(NULL,"\0");
        //printf("specail:%s %s\n",idx,val);

        x_space[temp].index = (int) strtol(idx,&endptr,10);
        x_space[temp].value = strtod(val,&endptr);

        if(inst_max_index > max_index)
            max_index = inst_max_index;

        x_space[j++].index = -1;
    }
    

    if(param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;

    fclose(fp);
}



