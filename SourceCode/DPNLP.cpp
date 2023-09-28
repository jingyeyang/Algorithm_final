#include<iostream>
#include<fstream> // for file input and output.
#include<vector>
#include<utility> // for pairs operation.
#include<iomanip> // for control the output format e.g. setw().
#include<map> // for map data structure => implement by red-black tree.
#include<random>
#include<iterator>
#include<set>
#include<cmath>
#include<algorithm> // for find().
#include<time.h>
//#include <boost/math/special_functions/gamma.hpp>
//#include <boost/math/distributions.hpp>

using namespace std;

//#define TRACE;
//#define DOEXP;


#define DISTANCE 6
#define STOPLPA 0.7
#define NodeDigit 4




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
    int* distance; // recording the distance between node and its core node.
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
    
/*
    cout << " THE DISTINCT COMM ID : " << endl;
    cout << "    {  ";
    for(auto iter = distinct_comm.begin() ; iter != distinct_comm.end() ; iter++)
    {
        cout << *iter << "  ";
    }
    cout << "}" << endl;
*/

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

/*
    cout << " Show the compress comm label of each node :: " << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << " ndoe " << setw(2) << node_id << " : "<<graph_info -> label[node_id] << endl;
    }
*/



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

/*
    cout << " Show the commuinitites :: " << endl;
    int total_connect = 0;
    for(int comm_id = 0 ; comm_id < num_comm ; comm_id++)
    { 
        cout << "    COMM  " << comm_id << " : " << endl;
        cout << "        ";
        for(auto comm_item = (communities + comm_id) -> member.begin() ; comm_item != (communities + comm_id) -> member.end() ; comm_item++)
        {
            cout << *comm_item << "  ";
        }
        cout << endl;
        total_connect += (communities + comm_id) -> inner_connect;
        total_connect += (communities + comm_id) -> inter_connect;
        cout << "     inner connect = " << (communities + comm_id) -> inner_connect << endl;
        cout << "     inter connect = " << (communities + comm_id) -> inter_connect << endl;
    }
    cout << "==================================================================>>>> total connect = " << total_connect << endl;
*/

    for(int comm_id = 0 ; comm_id < num_comm ; comm_id++)
    {
        modularity += (communities + comm_id) -> inner_connect - ((pow(((communities + comm_id) -> inner_connect + (communities + comm_id) ->inter_connect), 2)) / (2 * (graph_info -> num_edge)));
    }
    modularity = modularity / (2 * (graph_info -> num_edge));

    return modularity;
}

vector<vector<int>> reBuildDataset(string data_set, GraphInfo* graph_info)
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
        cerr << "DATASET FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    cout << "DATASET FILE OPEN SUCCESS !!!" << endl;

    // Getting dataset's informations.
    input_file >> temp;
    getline(input_file, dataset_infor);
    input_file >> temp;
    input_file >> graph_info -> num_edge;
    input_file >> graph_info -> num_node;
    input_file >> temp;
    //input_file >> temp;
    graph_info -> avg_degree = (float)(2*graph_info -> num_edge)/graph_info -> num_node;

    // Reading data.
    input_file >> read_node1;
    input_file >> read_node2;

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////// .......... WRITE LOG FILE .......... //////////////////////////////////////////////

void writeFile(string write_result_file, GraphInfo* graph_info)
{
    ofstream output_file;

    output_file.open(write_result_file, ios:: out);
    if(!output_file)
    {
        cerr << "LOG FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    cout << "LOG FILE OPEN SUCCESS !!!" << endl;

    output_file << "%" << " " << graph_info -> num_node << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        output_file << node_id << " " << graph_info -> label[node_id] << endl;
    }
    output_file.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////// .......... DPNLP ALGO .......... //////////////////////////////////////////////

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

bool considerPeripheralNode(GraphInfo* graph_info, vector<vector<int>> &graph, int target_node)
{
    //bool push_or_not = 0;
    map<int,int> label_count;
    int total_label_count = 0;
    float percentage_label = 0.0;

    for(int edge = 0 ; edge < graph[target_node].size() ; edge++)
    {
        label_count[graph_info -> label[graph[target_node][edge]]]++;
        total_label_count++;
    }
    percentage_label = ((float)label_count[graph_info -> label[target_node]] / (float)total_label_count);
    //cout << " Neighbor " << setw(2) << target_node << "    ";
    //cout << "percentage_label = " << percentage_label << "  ====> ";

    if(percentage_label < STOPLPA)
    {
        return(1);
    }
    else
    {
        return(0);
    }
}


int benfirScore(GraphInfo* graph_info, vector<vector<int>> &graph, int target_node)
{
    map<int,float> benfit_score;
    map<int,int> label_count;
    vector<int> label_candidate;
    //cout << " target node = " << setw(2) << target_node << endl;
    for(int edge = 0 ; edge < graph[target_node].size() ; edge++)
    {
        label_count[graph_info -> label[graph[target_node][edge]]]++;
        benfit_score[graph_info -> label[graph[target_node][edge]]] = 0;
    }
    for(auto count = label_count.begin() ; count != label_count.end() ; count++)
    {
        //cout << "      Label " << setw(2) << count -> first << " get " << count -> second << " times" << endl; 
    }

    map<int,int> sum_of_degree;
    map<int,int> dis_node_label;
    map<int,int> num_edge_node_label;

    for(int edge = 0 ; edge < graph[target_node].size() ; edge++)
    {
        dis_node_label[graph_info -> label[graph[target_node][edge]]] = 555;
    }

    for(int edge = 0 ; edge < graph[target_node].size() ; edge++)
    {
        sum_of_degree[graph_info -> label[graph[target_node][edge]]] += graph_info -> degree[graph[target_node][edge]];
        num_edge_node_label[graph_info -> label[graph[target_node][edge]]] ++;
    }
    for(int edge = 0 ; edge < graph[target_node].size() ; edge++)
    {
        
        //cout << "LABEL " << graph_info -> label[graph[target_node][edge]] << endl;
        //cout << "graph_info -> distance[" << graph[target_node][edge] << "] = " << graph_info -> distance[graph[target_node][edge]] << endl;
        //cout << "dis_node_label[" << graph_info -> label[graph[target_node][edge]] << "] = " << dis_node_label[graph_info -> label[graph[target_node][edge]]] << endl;
        if(graph_info -> distance[graph[target_node][edge]] < dis_node_label[graph_info -> label[graph[target_node][edge]]])
        {
            dis_node_label[graph_info -> label[graph[target_node][edge]]] = graph_info -> distance[graph[target_node][edge]] + 1; // bug target node = 14
                                                                                                                                    //Label  8 get 2 hops
                                                                                                                                    //Label 10 get 1 hops
        }
        //cout << "///...///" << endl;
        //cout << "graph_info -> distance[" << graph[target_node][edge] << "] = " << graph_info -> distance[graph[target_node][edge]] << endl;
        //cout << "dis_node_label[" << graph_info -> label[graph[target_node][edge]] << "] = " << dis_node_label[graph_info -> label[graph[target_node][edge]]] << endl;
        //cout << "////////////////////////////////////////////" << endl;
        
       //cout << graph[target_node][edge] << "  ";
    }
    //cout << endl;



    for(auto count = benfit_score.begin() ; count != benfit_score.end() ; count++)
    {
        count -> second = ((DISTANCE - dis_node_label[count -> first]) * ((graph_info -> avg_degree * num_edge_node_label[count -> first]) + sum_of_degree[count -> first]));
        count -> second = count -> second / DISTANCE;
    }

    for(auto count = benfit_score.begin() ; count != benfit_score.end() ; count++)
    {
        //cout << "      Label " << setw(2) << count -> first << " benfit score =  " << count -> second << " " << endl; 
    }

    pair<int,float> dominate_label(0,0);
    for(auto count = benfit_score.begin() ; count != benfit_score.end() ; count++)
    {
        if(count -> second > dominate_label.second)
        {
            dominate_label.first = count -> first;
            dominate_label.second = count -> second;
        }
    }

    //cout << " ~ DOMINATE LABEL :: " << dominate_label.first << "   WITH  BENFIT SCORE = " << dominate_label.second << endl;

    return(dominate_label.first);
}


void DPNLP(GraphInfo* graph_info, vector<vector<int>> &graph)
{
    cout << "//////////.......... START DOING DPNLP ALGO ..........//////////" << endl;
    //cout << ((3 * graph_info -> avg_degree) / 2) << endl;
    vector<int> core_node;
    vector<int> degree1_and_2;
    vector<int> node_without_deg1_and_2;
    //set<int> peripheral_node;
    vector<int> peripheral_node;
    vector<int> peripheral_node_temp;
    //float edge_devide_by_node = graph_info -> num_edge / graph_info -> num_node;
    float pow_edge_devide_by_node = pow(graph_info -> num_edge / graph_info -> num_node, 2);
    // Initial each node an unique label and the distance.
    graph_info -> label = (int*)malloc((graph_info -> num_node + 1) * sizeof(int));
    graph_info -> distance = (int*)calloc((graph_info -> num_node + 1), sizeof(int));
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        graph_info -> label[node_id] = node_id;
    }

#ifdef DOEXP

    cout << endl;
    cout << "     /////..... IGNORE DEG 1 AND DEG 2 ...../////" << endl;
    cout << endl;

#endif

    // Ignore nodes with degree1 and degree2 until finalizing.
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        //cout << "  ..... NODE " << setw(2) << node_id << " ..... " << endl;
        if(graph_info -> degree[node_id] == 2)
        {
            //cout << "    Pushing into degree1_and_2 cause the degree of node" << node_id << " is " << graph_info -> degree[node_id] << endl;
            degree1_and_2.push_back(node_id);
            auto temp = remove(graph[graph[node_id][0]].begin(), graph[graph[node_id][0]].end(), node_id);
            graph[graph[node_id][0]].erase(temp, graph[graph[node_id][0]].end());
            temp = remove(graph[graph[node_id][1]].begin(), graph[graph[node_id][1]].end(), node_id);
            graph[graph[node_id][1]].erase(temp, graph[graph[node_id][1]].end());
        }
        else if (graph_info -> degree[node_id] == 1)
        {
            //cout << "    Pushing into degree1_and_2 cause the degree of node" << node_id << " is " << graph_info -> degree[node_id] << endl;
            degree1_and_2.push_back(node_id);
            auto temp = remove(graph[graph[node_id][0]].begin(), graph[graph[node_id][0]].end(), node_id);
            graph[graph[node_id][0]].erase(temp, graph[graph[node_id][0]].end());     
        }
        else
        {
            //cout << "    Don't do anything" << endl;
            node_without_deg1_and_2.push_back(node_id);
        }
    }

/*
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
*/


#ifdef DOEXP
    cout << endl;
    cout << "     /////..... CREATE INITIAL STRUCT OF COMMUNITY ...../////" << endl;
    cout << endl;
#endif

    // shuffle the vector randomly
    std::random_device rd; // obtain a random seed from the OS
    std::mt19937 gen(rd()); // seed the generator
    shuffle(begin(node_without_deg1_and_2), end(node_without_deg1_and_2), gen);

    // Creating initial structure of comunities.
    for(auto iter = node_without_deg1_and_2.begin() ; iter != node_without_deg1_and_2.end() ; iter++)
    //for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        int node_id = *iter;
        //cout << "  ..... NODE " << setw(2) << node_id << " ..... " << endl;
        //if(graph_info -> degree[node_id] > 2)
        {
        // Start to find core nodes.
            // Condition 1
            if(graph_info -> degree[node_id] >= pow_edge_devide_by_node)
            { 
                // Condition 2's conpare base matric compute.
                float avg_neighbor_degree = 0;
                for(int neighbor_id = 0 ; neighbor_id < graph_info -> degree[node_id] ; neighbor_id++)
                {
                    avg_neighbor_degree += graph_info -> degree[graph[node_id][neighbor_id]];
                }
                avg_neighbor_degree = avg_neighbor_degree / graph_info -> degree[node_id];
                //cout << "    DEG LAGER THAN AVG.DEG" << endl;
                //cout << "    Deg(N(" << node_id << ")) = " << avg_neighbor_degree << endl;
                // Condition 2
                if(graph_info -> degree[node_id] >= avg_neighbor_degree)
                {
                    // Condition 3
                    if(((graph_info -> degree[node_id] + avg_neighbor_degree) / 2) >= ((3 * graph_info -> avg_degree) / 2))
                    {
                        //cout << "    It's a core node !!!!!" << endl;
                        core_node.push_back(node_id);

                        // Node that pass three conditions -> core node -> start to propagate label to its neighbors.
                        for(int edge = 0 ; edge < graph[node_id].size() ; edge++)
                        {
                            //cout << "    neighbor  " <<  setw(2) << graph[node_id][edge] << "  ----->";
                            if(graph_info -> degree[node_id] > graph_info -> degree[graph[node_id][edge]])
                            {
                                //if(graph_info -> label[graph[node_id][edge]] != node_id)
                                if(graph_info -> label[graph[node_id][edge]] != graph[node_id][edge]) ///!!!!!!////
                                {
                                    if(graph_info -> degree[graph_info -> label[graph[node_id][edge]]] > graph_info -> degree[node_id])
                                    {
                                        // Label remain.
                                        //cout << "  label reamin" << endl;
                                    }
                                    else
                                    {
                                        graph_info -> label[graph[node_id][edge]] = graph_info -> label[node_id];
                                        //cout << "  label change" << endl;
                                        graph_info -> distance[graph[node_id][edge]] = 1;
                                    }
                                }
                                else
                                {
                                    graph_info -> label[graph[node_id][edge]] = graph_info -> label[node_id];
                                    //cout << "  label change" << endl;
                                    graph_info -> distance[graph[node_id][edge]] = 1;
                                }
                            }
                            else
                            {
                                if(find(core_node.begin(), core_node.end(), graph[node_id][edge]) != core_node.end())
                                {
                                    // label diffuse.
                                    //cout << "  start to diffuse" << endl;
                                    for(int diffuse_edge = 0 ; diffuse_edge < graph[node_id].size() ; diffuse_edge++)
                                    {
                                        //graph_info -> label[graph[node_id][diffuse_edge]] = graph_info -> label[graph[node_id][edge]];
                                        graph_info -> label[graph[node_id][diffuse_edge]] = graph_info -> label[graph[node_id][edge]];
                                        if(find(graph[graph[node_id][diffuse_edge]].begin(), graph[graph[node_id][diffuse_edge]].end(), graph_info -> label[graph[node_id][edge]]) == graph[graph[node_id][diffuse_edge]].end() && graph[node_id][diffuse_edge] != graph_info -> label[graph[node_id][diffuse_edge]])
                                        {
                                            graph_info -> distance[graph[node_id][diffuse_edge]] = 2;
                                        }
                                    }
                                }
                                else
                                {
                                    // Label reamin.
                                    //cout << "  label reamin" << endl;
                                }
                            }
                        }
                    }
                    else
                    {
                        //cout << "    It's a peripheral node !!!!!" << endl;
                        //peripheral_node.insert(node_id);
                        peripheral_node.push_back(node_id);
                    }
                }
                else
                {
                    //cout << "    It's a peripheral node !!!!!" << endl;
                    //peripheral_node.insert(node_id);
                    peripheral_node.push_back(node_id);
                }
            }
            else
            {
                //cout << "    It's a peripheral node !!!!!" << endl;
                //peripheral_node.insert(node_id);
                peripheral_node.push_back(node_id);
            }
        }

    }

/*
    // TRACE show the degree1_and_2 nodes and core nodes.
    cout << endl;
    cout << "  ///// ..... DEGREE 1 AND 2 NODES ...... ///// " << endl;
    cout << "    { ";
    for(int node_id = 0 ; node_id < degree1_and_2.size() ; node_id++)
    {
        cout << degree1_and_2[node_id] << " ";
    }
    cout << "}" << endl;
    cout << "  ///// ..... CORE NODES ...... ///// " << endl;
    cout << "    { ";
    for(int node_id = 0 ; node_id < core_node.size() ; node_id++)
    {
        cout << core_node[node_id] << " ";
    }
    cout << "}" << endl;
    cout << "  ///// ..... PERIPHERAL NODES ...... ///// " << endl;
    cout << "    { ";
    for(auto node_id = peripheral_node.begin() ; node_id != peripheral_node.end() ; node_id++)
    {
        cout << *node_id << " ";
    }
    cout << "}" << endl;
*/
    //set<int> temp = peripheral_node;
    int deominate_label;

    // Label propagation and updating for peripheral nodes.
    //while(!temp.empty())
/*
        for(auto node_id = peripheral_node.begin() ; node_id != peripheral_node.end() ; node_id++)
        {

            deominate_label = benfirScore(graph_info, graph, *node_id);
            cout << " ~ DOMINATE LABEL :: " << deominate_label << endl;
            graph_info -> label[*node_id] = deominate_label;

        }
*/
#ifdef DOEXP
    cout << endl;
    cout << "     /////..... DOING LPA FOR PERIPHERAL NODES ...../////" << endl;
    cout << endl;
#endif
    //int iter = 0;
    //int max_iter = 5000;

    


    //while(!(peripheral_node.empty()))
    while(!(peripheral_node.empty()))
    {
        //cout << "size :: " << peripheral_node.size() << endl;
        //iter++;
        //cout << "       INERATION " << setw(4) << iter << endl;
        //cout << "          len  = " << setw(5) << peripheral_node.size() << endl;
        //int node_id = *select_randomly(peripheral_node.begin(), peripheral_node.end());
        //peripheral_node.erase(find(peripheral_node.begin(), peripheral_node.end(), node_id));
        
        //auto iter = select_randomly(peripheral_node.begin(), peripheral_node.end());
        //int node_id = *iter;
        //peripheral_node.erase(iter);


        // shuffle the vector randomly
        std::random_device rd; // obtain a random seed from the OS
        std::mt19937 gen(rd()); // seed the generator
        shuffle(begin(peripheral_node), end(peripheral_node), gen);

        //for(auto iter = peripheral_node.begin() ; iter != peripheral_node.end() ; iter++)
        for(int id = 0 ; id < peripheral_node.size() ; id++)
        {
            //cout << "hahaha" << endl;
            //cout << (&iter) << " : " << *(iter) << endl;
            //int node_id = (*iter);
            //peripheral_node.erase(iter);
            int node_id = peripheral_node[id];
            //cout << node_id << endl;
            auto iter = find(peripheral_node.begin(), peripheral_node.end(), node_id);
            peripheral_node.erase(iter);

            deominate_label = benfirScore(graph_info, graph, node_id);
                
            
            // Modify distance with new core node.
            int min_distance = 200;
            if(graph_info -> label[node_id] != deominate_label)
            {
                for(int edge = 0 ; edge < graph[node_id].size() ; edge++)
                {
                    //cout << graph[node_id][edge] << "  ";
                    if(graph_info -> label[graph[node_id][edge]] == deominate_label)
                    {
                        //cout << graph[node_id][edge] << "  ";
                        if(graph_info -> distance[graph[node_id][edge]] < min_distance)
                        {
                            min_distance = graph_info -> distance[graph[node_id][edge]];
                        }
                    }
                }
                //cout << endl;
                graph_info -> distance[node_id] = min_distance + 1;

                // Update label.
                graph_info -> label[node_id] = deominate_label;

                // push nodes into periheral set cause the nodes label changes.
                for(int edge = 0 ; edge < graph[node_id].size() ; edge++)
                {
                    bool push_or_not;
                    push_or_not = considerPeripheralNode(graph_info, graph, graph[node_id][edge]);
                    if(push_or_not)
                    {
                        //cout << "   push node " << graph[node_id][edge] << " into peripheral set " << endl;
                        //peripheral_node.insert(graph[node_id][edge]);
                        if((find(peripheral_node.begin(), peripheral_node.end(), graph[node_id][edge]) == peripheral_node.end()) && (find(peripheral_node_temp.begin(), peripheral_node_temp.end(), graph[node_id][edge]) == peripheral_node_temp.end()))
                        {
                            //peripheral_node.push_back(graph[node_id][edge]);
                            peripheral_node_temp.push_back(graph[node_id][edge]);
                        }
                    }
                    /*
                    else
                    {
                        //cout << " don't do anything " << endl;
                    }
                    */
                }
            }
        }
        //peripheral_node.clear();
        for(auto iter = peripheral_node_temp.begin() ; iter != peripheral_node_temp.end() ; iter++)
        {
            peripheral_node.push_back(*iter);
        }
        peripheral_node_temp.clear();
            
        
    }

#ifdef DOEXP

    cout << endl;
    cout << "     /////..... FINANLZING DEG 1 AND DEG 2 ...../////" << endl;
    cout << endl;

#endif
    // Finalizing the communities with degree1 and degree2.
    for(int node_id = 0 ; node_id < degree1_and_2.size() ; node_id++)
    {
        if(graph_info -> degree[degree1_and_2[node_id]] == 1)
        {
            graph_info -> label[degree1_and_2[node_id]] = graph_info -> label[graph[degree1_and_2[node_id]][0]];
            //graph_info -> distance[degree1_and_2[node_id]] = 1;
        }
        else
        {
            graph_info -> label[degree1_and_2[node_id]] = benfirScore(graph_info, graph, degree1_and_2[node_id]);
            //graph_info -> distance[degree1_and_2[node_id]] = 1;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////// .......... READ DATASET AND BUILD GRAPH .......... //////////////////////////////////////////////

vector<vector<int>> readFile(string data_set, GraphInfo* graph_info)
{
    ifstream input_file;   
    string dataset_infor;
    string temp;
    int read_node1;  // Storing the input nodes.
    int read_node2;

    //cout << STOPLPA << endl;
    // File opening.
    input_file.open(data_set, ios::in);
    if(!(input_file))
    {
        cerr << "DATASET FILE OPEN FAIL !!!\n";
        exit(-1);
    }
    cout << "DATASET FILE OPEN SUCCESS !!!" << endl;

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
        //graph_info -> start_node = read_node1;
        graph_info -> start_node = 1;
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int main(int argc, char* argv[], char* env[])
{// ./DPNLP.exe  SRCFile LOGFile GTFile EXPFile[enter]
    // SRCFile :: dataset/dataset.name.txt
    // LOGFile ::dataset_log/dataset.name_log.txt
    // GTFile :: dataset_gt/dataset.name.GT.txt
    // EXPFile :: exp_record/dataset.name_exp_record.txt


    GraphInfo* graph_info = (GraphInfo*)malloc(1 * sizeof(GraphInfo));
    vector<vector<int>> graph;

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
        cout << "USAGE: ./LPA.exe  SRCFile LOGFILE GTFile EXPFile[enter]" << endl;
        exit(1);
    }



    start = clock();

    DPNLP(graph_info, graph);

    end = clock();

    algo_time_record = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "          ///// ..... EXECUTION TIME ..... /////          "  << endl;
    cout << " READ BUILD GRAPH TIME TAKES :: " << fixed << read_build_graph_time_record << " sec " << endl;
    cout << " ALGO DPNLP TAKES :: " << fixed << algo_time_record << " sec " << endl;






    // Output the results.
    // Writing the label record to file [dataset.name]_log.txt.
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
    cout << " NMI = " << NMI << endl;
    cout << endl;





    cout << "///////////////..... WRITTING EXPERIMENT RESULTS ..... ////////////////" << endl << endl;
    // Writing the experiment record to file [dataset.name]_exp_record.txt
        // file struct ::
            // time_of_build_graph(sec)  time_of_algo(sec)    #comm   modularity  
    cout << "WRITTING THE EXP RESULT INTO EXP_RECORD FILE ." << endl;
    
    write_exp_record(argv[4], read_build_graph_time_record, algo_time_record, num_comm, modularity, NMI);

    cout << " WRITTING :: " << endl;
    cout << "    READ BUILD GRAPH TIME = " << read_build_graph_time_record << endl;
    cout << "    ALGO TIME             = " << algo_time_record << endl;
    cout << "    NUM COMMUNITY         = " << num_comm << endl;
    cout << "    MODULARITY            = " << modularity << endl;
    cout << "    NMI                   = " << NMI << endl;
    cout << endl;

    return 0;

}







/*
    Improve idea ::
        1. change map structure into unorder_map 
            - map using red balck tree such slower than unorder_map using hash table.
            - unorder_map must include header file :: <unorder_map>



        2. change the adjancy list into CSR format
            - the cons of CSR format -> can't insert and delete successfully
                 (solution) :: an improved CSR format called Packed CSR 
                    (reference the thesis) :: Packed Compressed Sparse Row: A Dynamic Graph Representation (IEEE)
*/




/* [TRACE] 


    Show the graph :: 

    cout << " Show the graph :: " << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << "    node " << node_id << " :  " << endl; 
        cout << "        ";
        for(int neighbor_id = 0 ; neighbor_id < graph[node_id].size() ; neighbor_id++)
        {
            cout << graph[node_id][neighbor_id] << "  ";
        }
        cout << endl;
    }   
    cout << endl << endl;











*/