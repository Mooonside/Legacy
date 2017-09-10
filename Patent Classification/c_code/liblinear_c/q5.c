#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>  
#include <pthread.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#define TRAIN_PTHREAD 5
#define PRED_PTHREAD 5

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
struct feature_node **test_x;
double *test_y;
double *pred_y;

static int (*info)(const char *fmt,...) = &printf;
//q3
int *random_choice_0,*random_choice_1,*random_choice_2,*random_choice_3; 
struct model ***model_list;
int split_positive;
int split_negative;
struct feature_node **x_dist[4];
int data_dist[4] = {0};
int split[4] = {10,20,10,1};
int size[4] = {0};
int test_num = 0;
//q5
pthread_t *thread_pool;

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
        // while(check_label(p)){
        //     p = strtok(NULL," ");
        // }
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
        }else if(prob.y[i] == 1){
            data_dist[1] += 1;
        }else if(prob.y[i] == 2){
            data_dist[2] += 1;
        }else if(prob.y[i] == 3){
            data_dist[3] += 1;
        }
        //printf("%d sample has label:%f \n",i,prob.y[i]);
        if(label == NULL) // empty line
            exit_input_error(i+1);
        

        // while(check_label(label)){
        //     label = strtok(NULL," ");
        // }
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
    //     printf("%d %f\n",x_space[k].index,x_space[k].value);
    // }

    fclose(fp);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i;
    void (*print_func)(const char*) = NULL;    // default printing to stdout

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


void read_test(FILE *input)
{
    int max_nr_attr = 64;
    double target_label;
    int n;
    int nr_feature=get_nr_feature(model_list[0][0]);
    if(model_list[0][0]->bias==0)
        n=nr_feature+1;
    else
        n=nr_feature;

    max_line_len = 10000;
    line = (char *)malloc(max_line_len*sizeof(char));

    while(readline(input)!=NULL){
        test_num ++;
    }
    rewind(input);

    test_y = Malloc(double,test_num);
    test_x = Malloc(struct feature_node*,test_num);

    int i = 0;
    
    while(readline(input) != NULL)
    {
        test_x[i] = Malloc(feature_node,max_nr_attr);
        int j = 0;
        char *idx, *val, *label, *endptr;
        int inst_max_index = 0; 

        label = strtok(line," ");
        if(label == NULL) 
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

        // while(check_label(label)){
        //     label = strtok(NULL," ");
        // }
        do{
            label = strtok(NULL," ");
        }while(check_label(label));

        j = 1;
        
        while(1)
        {
            if(j>=max_nr_attr-2) // need one more for index = -1
            {
                max_nr_attr *= 2;
                //x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
                test_x[i] = (struct feature_node *) realloc(test_x[i],max_nr_attr*sizeof(struct feature_node));
            }
            
            idx = strtok(NULL,":");
            val = strtok(NULL," ");

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
            
            if(test_x[i][j].index <= nr_feature)
                ++j;
        }

        idx= strtok(label,":");
        val = strtok(NULL,"\0");

        test_x[i][0].index = (int) strtol(idx,&endptr,10);
        test_x[i][0].value = strtod(val,&endptr);

        if(model_list[0][0]->bias>=0)
        {
            test_x[i][j].index = n;
            test_x[i][j].value = model_list[0][0]->bias;
            j++;
        }
        test_x[i][j].index = -1;
        ++i;
    }
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
                        int p1,int size1,struct feature_node** x2,int p2,int size2)
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


void *thread_train(void *thread_param){
    int step = split[0] / TRAIN_PTHREAD;
    int start = *(int *) thread_param;
    int upper = step*(start+1);
    if(start == TRAIN_PTHREAD -1){
       upper = split_positive;
    }

    struct problem mini_prob;
    struct model* mini_model;

    for (int it = step*start;it < upper;it++){

        mini_prob.n = prob.n;
        mini_prob.bias = prob.bias;

        for(int it1 = 0;it1<split[1];it1++){
            mini_prob.l = size[0] + size[1];
            mini_prob.y = Malloc(double,mini_prob.l);
            mini_prob.x = Malloc(struct feature_node*,mini_prob.l);
            mini_maker(mini_prob,x_dist[0],it*size[0],size[0],
                x_dist[1],it1*size[1],size[1]);
            mini_model = train(&mini_prob, &param);
            //printf("model %d,%d is trained\n",it,it1);
            model_list[it][it1] = mini_model;
            free(mini_prob.y);
            free(mini_prob.x);
        }

        for(int it1 = 0;it1<split[2];it1++){
            mini_prob.l = size[0] + size[2];

            mini_prob.y = Malloc(double,mini_prob.l);
            mini_prob.x = Malloc(struct feature_node*,mini_prob.l);
            mini_maker(mini_prob,x_dist[0],it*size[0],size[0],
                x_dist[2],it1*size[2],size[2]);
            mini_model = train(&mini_prob, &param);
            model_list[it][split[1]+it1] = mini_model;
            //printf("model %d,%d is trained\n",it,it1+split[1]);
            free(mini_prob.y);
            free(mini_prob.x);
        }

        for(int it1 = 0;it1<split[3];it1++){
            mini_prob.l = size[0] + size[3];

            mini_prob.y = Malloc(double,mini_prob.l);
            mini_prob.x = Malloc(struct feature_node*,mini_prob.l);
            mini_maker(mini_prob,x_dist[0],it*size[0],size[0],
                x_dist[3],it1*size[3],size[3]);
            mini_model = train(&mini_prob, &param);
            model_list[it][split[1]+split[2]+it1] = mini_model;
            //printf("model %d,%d is trained\n",it,it1+split[1]+split[2]);
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
                predict_values(model_list[it][it1],test_x[i],&temp);
                if(temp < min){
                    min = temp;
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

    parse_command_line(argc, argv, input_file_name, test_file_name);
    printf("Train data:%s\n",input_file_name);
    printf("Test data:%s\n",test_file_name);

    read_problem(input_file_name);
    
    // now do data sepratation
    printf("%d\n",prob.l); 
    printf("data distribution: %d,%d,%d,%d\n",data_dist[0],data_dist[1],data_dist[2],data_dist[3]);


    for (int it=0;it<4;it++){
        x_dist[it] = Malloc(struct feature_node *,data_dist[it]);
    }

    int temp_it[4] = {0};
    for (int it = 0;it<prob.l;it++){
        if(prob.y[it] == 0){
            x_dist[0][temp_it[0]] = prob.x[it];             
            temp_it[0] ++;
        }else if(prob.y[it] == 1){
            x_dist[1][temp_it[1]] = prob.x[it];
            temp_it[1] ++;
        }else if(prob.y[it] == 2){
            x_dist[2][temp_it[2]] = prob.x[it];
            temp_it[2] ++;     
        }else if(prob.y[it] == 3){
            x_dist[3][temp_it[3]] = prob.x[it];
            temp_it[3] ++;
        }
    }


    random_choice_0 = Malloc(int,data_dist[0]);
    random_choice_1 = Malloc(int,data_dist[1]);
    random_choice_2 = Malloc(int,data_dist[2]);
    random_choice_3 = Malloc(int,data_dist[3]);
    shuffle(random_choice_0,data_dist[0]);
    shuffle(random_choice_1,data_dist[1]);
    shuffle(random_choice_2,data_dist[2]);
    shuffle(random_choice_3,data_dist[3]);

    split_positive = split[0];
    split_negative = 0;
    for (int it=1;it<4;it++){
        split_negative += split[it];
    }
    printf("##########################################\n");
    printf("we got %d models to train!\n",split_positive*split_negative);
    printf("##########################################\n");

    for (int it=0;it<4;it++){
        size[it] = int(data_dist[it]/split[it]);
    }

    printf("##########################################\n");
    printf("mini problem size:\n");
    for(int it=0;it<4;it++){
        printf("%d ",size[it]);
    }
    printf("\n");
    printf("##########################################\n");

    clock_gettime(CLOCK_MONOTONIC,&tic);

    model_list = Malloc(struct model**,split_positive);
    for (int it=0;it < split_positive;it++){
        model_list[it] = Malloc(struct model*,split_negative);
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
    printf("Successfully trained\n");
    printf("##########################################\n");

    char output_file_name[] = "linear_pred_prior"; 
    
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
        if(pred_y[i]>=0){
            temp = 1;
        }else{
            temp = -1;
        }
        if(temp == test_y[i]){
            correct++;
        }
    }
    info("Accuracy = %g%% (%d/%d)\n",(double) correct/test_num*100,correct,test_num);

    clock_gettime(CLOCK_MONOTONIC,&tiiic);

    double train_time = tiic.tv_sec - tic.tv_sec;
    train_time += (tiic.tv_nsec - tic.tv_nsec)/1000000000.0;
    
    double pred_time = tiiic.tv_sec - tiic.tv_sec;
    pred_time += (tiiic.tv_nsec - tiic.tv_nsec)/1000000000.0;
    
    printf("train time : %f s\n",train_time);
    printf("pred time : %f s\n",pred_time);
}