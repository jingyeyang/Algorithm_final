#include<iostream>
#include<fstream> // for file input and output.
#include<vector>
#include<utility> // for pairs operation.
#include<iomanip> // for control the output format e.g. setw().
#include<map> // for map data structure => implement by red-black tree.
#include<random>
#include<iterator>
#include<set>
#include<algorithm>
#include<random>

using namespace std;

//#define TRACE;
//#define DOEXP;
#define MAX_ITERATION 1000

typedef struct Dataset_info
{
    int num_node;
    int start_node;
    int end_node;
    int* algo_label;
    //int* gt_label;
    int algo_num_comm;
    int gt_numn_comm;
    set<int> algo_comm_id;
    set<int> gt_comm_id;
    map<int,int> gt_label;

}Dataset_info;


typedef struct GraphInfo
{
    int num_edge;
    int num_node;
    int start_node;
    int end_node;
    float avg_degree;
    int* degree;
    int* label;
}GraphInfo;

typedef struct Community
{
    set<int> member;
    int inner_connect = 0;
    int inter_connect = 0;
}Community;




void write_exp_record(string write_exp_record_file, double build_graph_time, double algo_time, int num_community, float modularity, float NMI)
{
    ofstream output_file;
    output_file.open(write_exp_record_file, ios:: app);
    if(!output_file)
    {
        cerr << "EXP_RECOED FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    //cout << "EXP_RECOED FILE OPEN SUCCESS !!!" << endl;

    output_file << build_graph_time << "," << algo_time << "," << num_community  << "," << modularity << "," << NMI << endl;

    output_file.close();
}

///////////////////////////////////// .......... NMI .......... //////////////////////////////////////////////

float computeMutualInformation(Dataset_info* dataset, vector<set<int>> &algo_partition, vector<set<int>> &gt_partition)
{
    // Compute MI. 
    set<int> intersection_temp;
    float N = dataset -> num_node;
    float MI = 0.0;

    //cout << "GETTING ITERSECTIONS :: " << endl;;
    for(int algo_comm_id = 0 ; algo_comm_id < algo_partition.size() ; algo_comm_id++)
    {
        for(int gt_comm_id = 0 ; gt_comm_id < gt_partition.size() ; gt_comm_id++)
        {  
            intersection_temp.clear();

            set_intersection(algo_partition[algo_comm_id].begin(), algo_partition[algo_comm_id].end(), gt_partition[gt_comm_id].begin(), gt_partition[gt_comm_id].end(), inserter(intersection_temp, intersection_temp.begin()));
            
/*
            cout<< "  algo_partition's" << setw(NodeDigit) << algo_comm_id << "   n  gt_partition's" << setw(NodeDigit) << gt_comm_id << endl;
            cout << "    {  ";
            for(auto iter = intersection_temp.begin() ; iter != intersection_temp.end() ; iter++)
            {
                cout << *iter << "  ";
            }
            cout << "}" << endl;
            cout << "    ----- size = " << intersection_temp.size() << endl;
*/

            float num_itersection = intersection_temp.size();
            //float N = dataset -> num_node;
            float num_X = algo_partition[algo_comm_id].size();
            float num_Y = gt_partition[gt_comm_id].size();
            float p_x_y = (num_itersection / N);
            float p_x_y_devide_px_py = ((num_itersection * N) / (num_X * num_Y));
            float log_p_x_y_devide_px_py = log(p_x_y_devide_px_py);
            float total = (p_x_y * log_p_x_y_devide_px_py);
/*
            //cout<< "  algo_partition's" << setw(NodeDigit) << algo_comm_id << "   n  gt_partition's" << setw(NodeDigit) << gt_comm_id << endl;
            cout << "    P(X,Y) = " << p_x_y << endl;
            cout << "    [P(X,Y) / P(X)P(Y)] = " << p_x_y_devide_px_py << endl;
            cout << "    log[P(X,Y) / P(X)P(Y)] = " << log_p_x_y_devide_px_py << endl;
            cout << "    ----- TOTAL = " << total << endl;
*/
            if(p_x_y > 0)
            {
                MI += total;
            }
            //MI += total;
        }
    }
    cout << " METUAL INFORMATION :: " << MI << endl;
    return MI;
}

float computeNormalizedMutualInformation(Dataset_info* dataset, vector<set<int>> &algo_partition, vector<set<int>> &gt_partition)
{
    float NMI = 0.0;
    float H_X = 0.0;
    float H_Y = 0.0;
    float N = dataset -> num_node;

    for(int algo_comm_id = 0 ; algo_comm_id < algo_partition.size() ; algo_comm_id++)
    {
        float num_X = algo_partition[algo_comm_id].size();
        float P_X = (num_X / N);
        float log_PX = log(P_X);
        float total = (P_X * log_PX);
        H_X -= total; 
    }
    for(int gt_comm_id = 0 ; gt_comm_id < gt_partition.size() ; gt_comm_id++)
    {
        float num_Y = gt_partition[gt_comm_id].size();
        float P_Y = (num_Y / N);
        float log_PY = log(P_Y);
        float total = (P_Y * log_PY);
        H_Y -= total; 
    }

    //cout << " H(algo_partition) = " << H_X << endl;
    //cout << " H(gt_partition) = " << H_Y << endl;
    return (2 * computeMutualInformation(dataset, algo_partition, gt_partition)) / (H_X + H_Y);
}

void buildPartition(Dataset_info* dataset, vector<set<int>> &algo_partition, vector<set<int>> &gt_partition)
{
    int group_id = 0;
    for(auto iter = dataset -> algo_comm_id.begin() ; iter != dataset -> algo_comm_id.end() ; iter++)
    {
        for(int node_id = dataset -> start_node ; node_id <= dataset -> end_node ; node_id++)
        {
            if(dataset -> algo_label[node_id] == *iter)
            {
                algo_partition[group_id].insert(node_id);
            }
        }
        group_id++;
    }

/*
    group_id = 0;
    for(auto iter = dataset -> gt_comm_id.begin() ; iter != dataset -> gt_comm_id.end() ; iter++)
    {
        for(int node_id = dataset -> start_node ; node_id <= dataset -> end_node ; node_id++)
        {
            if(dataset -> gt_label[node_id] == *iter)
            {
                gt_partition[group_id].insert(node_id);
            }
        }
        group_id++;
    }
*/


    group_id = 0;
    for(auto comm_id = dataset -> gt_comm_id.begin() ; comm_id != dataset -> gt_comm_id.end() ; comm_id++)
    {
        for(auto label_id = dataset -> gt_label.begin() ; label_id != dataset -> gt_label.end() ; label_id++)
        {
            if(((*label_id).second) == *comm_id)
            {
                int node_id = (*label_id).first; 
                //cout << node_id << "  with  " << (*label_id).second << endl;
                gt_partition[group_id].insert(node_id);
            }
        }
        group_id++;
    }
}

void partitionDataset(Dataset_info* dataset)
{


    for(int node_id = dataset -> start_node ; node_id <= dataset -> end_node ; node_id++)
    {
        dataset -> algo_comm_id.insert(dataset -> algo_label[node_id]);
        //dataset -> gt_comm_id.insert(dataset -> gt_label[node_id]);
    }
    for(auto iter = dataset -> gt_label.begin() ; iter != dataset -> gt_label.end() ; iter++)
    {
        dataset -> gt_comm_id.insert((*iter).second);
    }

    dataset -> algo_num_comm = dataset -> algo_comm_id.size();
    dataset -> gt_numn_comm = dataset -> gt_comm_id.size();

    //cout << dataset -> algo_num_comm << endl;
    //cout << dataset -> gt_numn_comm << endl;
}

void readGT(string algo_partition_result, string gt_partition_result, Dataset_info* dataset)
{
    ifstream algo_result;   
    ifstream gt_result;

    int node;
    int comm;
    string temp;

    // File opening.
    algo_result.open(algo_partition_result, ios::in);
    gt_result.open(gt_partition_result, ios :: in);
    if(!(algo_result) || (!gt_result))
    {
        cerr << "PARTITION RESULT FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    //cout << "PARTITION RESULT FILE OPEN SUCCESS !!!" << endl;

    algo_result >> temp;
    algo_result >> dataset -> num_node;


    dataset -> algo_label = (int*)calloc(dataset -> num_node + 1, sizeof(int));
    //dataset -> gt_label = (int*)calloc(dataset -> num_node + 1, sizeof(int));


    algo_result >> node;
    algo_result >> comm;
    if(node == 0)
    {
        dataset -> start_node  = node;
        dataset -> end_node = dataset -> num_node - 1;             // MUST MODIFY !!! CAUSE THE NODE START IN 0 AND END IN NUM_NODE-1 !!!
    }
    else
    {
        dataset -> start_node = 1;
        dataset -> end_node = dataset -> num_node; 
    }
    dataset -> algo_label[node] = comm;

    while(!algo_result.eof())
    {
        algo_result >> node;
        algo_result >> comm;
        dataset -> algo_label[node] = comm; 
    }
    while(!gt_result.eof())
    {
        gt_result >> node;
        gt_result >> comm;
        //dataset -> gt_label[node] = comm; 
        dataset -> gt_label.insert(pair<int,int>(node, comm));
    }

    algo_result.close();
    gt_result.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////// .......... MODULARITY .......... ////////////////////////////////////////////////

float modularityCompute(GraphInfo* graph_info, int num_comm, set<int> distinct_comm, vector<vector<int>> &graph)
{
    float modularity = 0.0;
    // Storing real community label corresponding to the order
    int* correspond_comm_id = (int*)malloc(num_comm * sizeof(int));
    if(!correspond_comm_id) cout << " FAIL MALLOC !!!" << endl;
    
    int comm_id = 0;
    for(auto iter = distinct_comm.begin() ; iter != distinct_comm.end() ; iter++)
    {
        //*(correspond_comm_id + comm_id) = *iter;
        //comm_id++;
        for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
        {
            if((graph_info -> label[node_id]) == (*iter))
            {
                graph_info -> label[node_id] = comm_id;
            }
        }
        comm_id++;
    }

    // Malloc the communtiy structure for computing modularity
    Community* communities = new Community [num_comm];
    if(!communities) cout << " FAIL NEW !!!" << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        (communities + (graph_info -> label[node_id])) -> member.insert(node_id);
        for(int neighbor_id = 0 ; neighbor_id < graph[node_id].size() ; neighbor_id++)
        {
            if(node_id <= graph[node_id][neighbor_id])
            {
                if((graph_info -> label[node_id]) == ((graph_info) -> label[graph[node_id][neighbor_id]]))
                {
                    (communities + (graph_info -> label[node_id])) -> inner_connect += 2;
                }
                else
                {
                    (communities + (graph_info -> label[node_id])) -> inter_connect++;
                    (communities + (graph_info -> label[graph[node_id][neighbor_id]])) -> inter_connect++;
                }
            }
        }
    }

    for(int comm_id = 0 ; comm_id < num_comm ; comm_id++)
    {
        modularity += (communities + comm_id) -> inner_connect - ((pow(((communities + comm_id) -> inner_connect + (communities + comm_id) ->inter_connect), 2)) / (2 * (graph_info -> num_edge)));
    }
    modularity = modularity / (2 * (graph_info -> num_edge));

    return modularity;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////// .......... WRITE LOG FILE .......... //////////////////////////////////////////////

void writeFile(string write_result_file, GraphInfo* graph_info)
{
    ofstream output_file;

    output_file.open(write_result_file, ios:: out);
    if(!output_file)
    {
        cerr << "FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    cout << "FILE OPEN SUCCESS !!!" << endl;

    output_file << "%" << " " << graph_info -> num_node << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        output_file << node_id << " " << graph_info -> label[node_id] << endl;
    }
    output_file.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}


template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

void LPA(GraphInfo* graph_info, vector<vector<int>> &graph, vector<int> node_sequence)
{
    cout << "//////////.......... START DOING LPA ALGO ..........//////////" << endl;
    // Initial each node an unique label.
    graph_info -> label = (int*)malloc((graph_info -> num_node + 1) * sizeof(int));
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        graph_info -> label[node_id] = node_id;
    }

    // Start to label propagation.
    map<int,int> label_count;
    vector<int> label_candidate;
    int flag = 1;
    int iteration = 0;
    //int max_iteration = 10000;
    while(flag && iteration < MAX_ITERATION)
    {
        flag = 0;
        iteration++;

#ifdef TRACE
        cout << "   /////..... ITERATION " << iteration << " ...../////" << endl;
#endif
        // shuffle the vector randomly
        std::random_device rd; // obtain a random seed from the OS
        std::mt19937 gen(rd()); // seed the generator
        shuffle(begin(node_sequence), end(node_sequence), gen);


        // For all nodes select the it's neighbors label.
        //for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
        for(auto node_iter = node_sequence.begin() ; node_iter != node_sequence.end() ; node_iter++)
        {
            int node_id = (*node_iter);
            // Choose the largest #label as the dominate label, if have same #label than randomly select one.
#ifdef TRACE
            cout << "    NODE " << setw(2) << node_id << endl;
#endif
            //map<int,int> label_count;
                // Count #label arround the node.
            for(auto label_id = graph[node_id].begin() ; label_id != graph[node_id].end() ; label_id++)
            {
                    label_count[graph_info -> label[*label_id]]++;
            }

#ifdef TRACE
            for(auto count = label_count.begin() ; count != label_count.end() ; count++)
            {
                cout << "      Label " << setw(2) << count -> first << " get " << count -> second << " times" << endl; 
            }
#endif

                // Find the maximum #label arround the node.
            int max_num_label = 0;
            for(auto count = label_count.begin() ; count != label_count.end() ; count++)
            {
                if(count -> second > max_num_label) max_num_label = count -> second;
            }

#ifdef TRACE
            cout << "          MAX #Label = " << setw(2) << max_num_label << endl;
#endif      
                // Push the labels with max num into vector.
            //vector<int> label_candidate;
            for(auto count = label_count.begin() ; count != label_count.end() ; count++)
            {
                if(count -> second == max_num_label) label_candidate.push_back(count -> first);
            }
                // Randomly choose a label as dominate label.
            int dominate_label = *select_randomly(label_candidate.begin(), label_candidate.end());

#ifdef TRACE
            cout << "          .....Dominate label = " << dominate_label << endl;
#endif
            
            if(graph_info -> label[node_id] != dominate_label)
            {
                graph_info -> label[node_id] = dominate_label;
                // If there is no change of labels than stop.
                flag = 1;
            }

            label_candidate.clear();
            label_count.clear();
        }
    }

    //cout << iteration << endl;
    //return(iteration);
}


vector<vector<int>> readFile(string data_set, GraphInfo* graph_info)
{
    ifstream input_file;   
    string dataset_infor;
    string temp;
    int read_node1;  // Storing the input nodes.
    int read_node2;

    // File opening.
    input_file.open(data_set, ios::in);
    if(!(input_file))
    {
        cerr << "FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    cout << "FILE OPEN SUCCESS !!!" << endl;

    // Getting dataset's informations.
    input_file >> temp;
    getline(input_file, dataset_infor);
    input_file >> temp;
    input_file >> graph_info -> num_edge;
    input_file >> graph_info -> num_node;
    input_file >> temp;
    //input_file >> temp;
    graph_info -> avg_degree = (float)(2*graph_info -> num_edge)/graph_info -> num_node;
        // Print dataset's informations.
    cout << "==================================" << endl;
    cout << "       DATASET INFORMATIONS       " << endl;
    cout << ".................................." << endl;
    cout << "    NAME       = " << data_set << endl;
    cout << "    BASIC INFO = " << dataset_infor << endl;
    cout << "    #EDGE      = " << graph_info -> num_edge << endl;
    cout << "    #NODE      = " << graph_info -> num_node << endl;
    cout << "    AvgDegree  = " << graph_info -> avg_degree << endl;
    cout << "==================================" << endl;

    // Reading data.
    input_file >> read_node1;
    input_file >> read_node2;

#ifdef TRACE
    cout << "read node 1 = " << read_node1 << endl;
    cout << "read node 2 = " << read_node2 << endl;
#endif
    
        // Consider the beginning of dataset is 0 or 1.
    if(read_node1 == 0)
    {
        graph_info -> start_node  = read_node1;
        graph_info -> end_node = graph_info -> num_node - 1;             // MUST MODIFY !!! CAUSE THE NODE START IN 0 AND END IN NUM_NODE-1 !!!
    }
    else
    {
        graph_info -> start_node = read_node1;
        graph_info -> end_node = graph_info -> num_node; 
    }
    // Create a space to store graph and it's degree.
    vector<vector<int>> graph(graph_info -> num_node + 1);
    graph_info -> degree = (int*)calloc((graph_info -> num_node + 1), sizeof(int));


    // Start buliding graph into adjancey list.
    graph[read_node1].push_back(read_node2);
    graph_info -> degree[read_node1]++;
    graph[read_node2].push_back(read_node1);
    graph_info -> degree[read_node2]++;
    while(!input_file.eof())
    {
        input_file >> read_node1;
        input_file >> read_node2;
        //cout << read_node1 << " " << read_node2 << endl;
        graph[read_node1].push_back(read_node2);
        graph_info -> degree[read_node1]++;
        graph[read_node2].push_back(read_node1);
        graph_info -> degree[read_node2]++;
    }
    
    input_file.close();
    return(graph);
}


int main(int argc, char* argv[], char* env[])
{// ./LPA_random_version.exe  SRCFile LOGFile GTFile EXPFile[enter]
    // SRCFile :: dataset/dataset.name.txt
    // LOGFile ::dataset_log/dataset.name_log.txt
    // GTFile :: dataset_gt/dataset.name.GT.txt
    // EXPFile :: exp_record/dataset.name_exp_record.txt


    GraphInfo* graph_info = (GraphInfo*)malloc(1 * sizeof(GraphInfo));
    vector<vector<int>> graph;
    vector<int> node_sequence;

    // Recording execution time
    clock_t start, end;
    double read_build_graph_time_record;
    double algo_time_record;

    if(argc == 5)
    {
        clock_t start, end;

        start = clock();
        graph = (vector<vector<int>>) readFile(argv[1], graph_info);
        end = clock();

        read_build_graph_time_record = (double)(end - start) / CLOCKS_PER_SEC;
    }
    else
    {
        cout << "USAGE: ./LPA_random_version.exe  SRCFile LOGFile GTFile EXPFile[enter]" << endl;
        exit(1);
    }
 
 #ifdef TRACE
    // Print the graph.
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << "NODE " << node_id << "'s edges : " << endl;  
        for(int edge = 0 ; edge <= graph[node_id].size() - 1 ; edge++)
        {
            cout << setw(3) << graph[node_id][edge] << endl;
        }
        cout << "  DEGREE(" << setw(2) << node_id << ") = " << graph_info -> degree[node_id] << endl; 
    }
#endif

    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        node_sequence.push_back(node_id);
    }

    int iteration;

    //cout << "//////////.......... START DOING LPA ALGO ..........//////////" << endl;
    start = clock();

    //iteration = LPA(graph_info, graph, node_sequence);
    LPA(graph_info, graph, node_sequence);

    end = clock();

    algo_time_record = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "          ///// ..... EXECUTION TIME ..... /////          "  << endl;
    cout << " READ BUILD GRAPH TIME TAKES :: " << fixed << read_build_graph_time_record << " sec " << endl;
    cout << " ALGO DPNLP TAKES :: " << fixed << algo_time_record << " sec " << endl;

/*
    // print #iterations.
    cout << "////////// .......... AFTER DOING LPA IN " << setw(3) << iteration << " ITERATION .......... //////////" << endl; 


    // Print label.
    cout << "          ///// ..... LABEL ..... /////          " << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << "            NODE " << setw(2) << node_id << " = " << graph_info -> label[node_id] << endl;
    }
*/
    // Output the results.
    cout << "WRITTING THE RESULT INTO LOG FILE ." << endl;
    writeFile(argv[2], graph_info);
    cout << "FINISH WRITTING !!! " << endl;




    // Caculating the measurements.
    cout << "///////////////..... EXPERIMENT MEASUREMENTS ..... ////////////////" << endl << endl;

    map<int,int> label_count;
    set<int> distinct_comm; // For storing the diferent number of community id.
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        label_count[graph_info -> label[node_id]]++;
    }

    int num_comm = label_count.size();
    cout << " TOTAL #COMMUNITY =  " << num_comm << endl; 

    for(auto map_id = label_count.begin() ; map_id != label_count.end() ; map_id++)
    {
        distinct_comm.insert((*map_id).first);
    }

    float modularity;
    modularity = modularityCompute(graph_info, num_comm, distinct_comm, graph);
    cout << " MODULARITY = " << modularity << endl;

    // Compute NMI.
     Dataset_info* dataset = new Dataset_info[sizeof(Dataset_info)];

        // Read SRCFile and GTFile.
    readGT(argv[2], argv[3], dataset);

        // Partition the dataset into groups.
    partitionDataset(dataset);

    vector<set<int>> algo_partition(dataset -> algo_num_comm);
    vector<set<int>> gt_partition(dataset -> gt_numn_comm);

    buildPartition(dataset, algo_partition, gt_partition);

        // Compute NMI.
    float NMI;
    NMI = computeNormalizedMutualInformation(dataset, algo_partition, gt_partition);
    //cout << " NMI = " << NMI << endl;
    cout << endl;





    cout << "///////////////..... WRITTING EXPERIMENT RESULTS ..... ////////////////" << endl << endl;
    // Writing the experiment record to file [dataset.name]_exp_record.txt
        // file struct ::
            // time_of_build_graph(sec)  time_of_algo(sec)    #comm   modularity  
    cout << "WRITTING THE EXP RESULT INTO EXP_RECORD FILE ." << endl;
    
    write_exp_record(argv[4], read_build_graph_time_record, algo_time_record, num_comm, modularity, NMI);

    cout << " WRITTING :: " << endl;
    cout << "    READ BUILD GRAPH TIME = " << read_build_graph_time_record << "sec" << endl;
    cout << "    ALGO TIME             = " << algo_time_record << "sec" << endl;
    cout << "    NUM COMMUNITY         = " << num_comm << endl;
    cout << "    MODULARITY            = " << modularity << endl;
    cout << "    NMI                   = " << NMI << endl;
    cout << endl;

    return 0;

}



