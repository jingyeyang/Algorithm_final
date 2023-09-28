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
#include<cstring>

using namespace std;

//#define DOEXP;
#define node_digit 2


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


typedef struct FinalCommunity
{
    set<int> member;
    int inner_connect = 0;
    int inter_connect = 0;
}FinalCommunity;


typedef struct GraphInfo
{
    int num_edge;
    int num_node;
    int start_node;
    int end_node;
    float avg_degree;
    int* degree;
    int* label;
    //int* distance; // recording the distance between node and its core node.
    int num_community;
    int start_community;
    int end_community;
    // Using in the Louvain iterations.
    //int original_num_node;
    int original_start_node;
    int original_end_node;
    int* original_degree;
    int* original_label;
    int* loop;
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
    for(int node_id = graph_info -> original_start_node ; node_id <= graph_info -> original_end_node ; node_id++)
    {
        output_file << node_id << " " << graph_info -> original_label[node_id] << endl;
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


float modularityCompute(Community* communities, int num_edge, int start_comm, int end_comm)
{
    float modularity = 0.0;
    for(int comm_id = start_comm ; comm_id <= end_comm ; comm_id++)
    {
        modularity += (communities + comm_id) -> inner_connect - ((pow(((communities + comm_id) -> inner_connect + (communities + comm_id) ->inter_connect), 2)) / (2 * num_edge));
    }

    modularity = modularity / (2 * num_edge);

    return modularity;
}

float deltaModularity(Community* comm_moving_in, Community* comm_moving_out, int moving_node_degree ,int k_in, int num_edge, int loop)
{
    float moving_after;
    float moving_before;
    float delta_modularity;



    float loop_f = loop;

    moving_after = (comm_moving_in -> inner_connect + (2 * k_in) + loop_f) - (pow(comm_moving_in -> inner_connect + comm_moving_in -> inter_connect + (moving_node_degree) + loop_f , 2) / (2 * num_edge));

    moving_before = (comm_moving_in -> inner_connect) - ((pow(comm_moving_in -> inner_connect + comm_moving_in -> inter_connect, 2) / (2 * num_edge))) + (comm_moving_out -> inner_connect)- ((pow(comm_moving_out -> inner_connect + comm_moving_out -> inter_connect, 2)) / (2 * num_edge));


    delta_modularity = (moving_after - moving_before) / (2 * num_edge);


    return delta_modularity;
}




float Louvain(GraphInfo* graph_info, vector<vector<int>> &graph)
{
    cout << "//////////.......... START DOING LOUVAIN ALGO ..........//////////" << endl;

    map<int,int> neighbor_community; // Storing <comm_id, inner_connect between node and comm_id>
    float delta_modularity;
    float max_delta_modularity;
    int change_label;
    int flag_phase_1 = 1;
    FinalCommunity* final_communities = NULL;
    Community* temp_1;
    FinalCommunity* temp_2;
    int num_comm = graph_info -> num_node;
    float max_modularity = 0.0;



    vector<float> mod_record;
    //float* mod_record = (float*)calloc(20, sizeof(float));


    // Initial 
#ifdef DOEXP  
    cout << "INITIAL THE COMMUNITY .......... " << endl;
#endif
        // Label
    graph_info -> label = (int*)malloc((graph_info -> num_node + 1) * sizeof(int));
    graph_info -> original_label = (int*)malloc((graph_info -> num_node + 1) * sizeof(int));
    
        // Loop
    graph_info -> loop = (int*)malloc((graph_info -> num_node + 1) * sizeof(int));

        // Community
    Community* communities = new Community[(graph_info -> num_node + 1)];
    FinalCommunity* temp_comunities = new FinalCommunity[(graph_info -> num_node + 1)];
    graph_info -> start_community = graph_info -> start_node;
    graph_info -> end_community = graph_info -> end_node;
    graph_info -> num_community = graph_info -> num_node;


    if(!graph_info -> label || !communities /*|| !graph_info -> original_label*/)
    {
        cout << " ERROR : initial malloc or new  fail !!! " << endl;
    }

    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        graph_info -> label[node_id] = node_id;
        graph_info -> original_label[node_id] = node_id;
        (communities + node_id) -> member.insert(node_id);
        (temp_comunities + node_id) -> member.insert(node_id);
        (communities + node_id) -> inner_connect = 0;
        (communities + node_id) -> inter_connect = graph[node_id].size();
        graph_info -> loop[node_id] = 0;

    }

    float modularity_begin = modularityCompute(communities, graph_info -> num_edge, graph_info -> start_community, graph_info -> end_community);
    float modularity = modularity_begin;
#ifdef DOEXP
    cout << "modularity = " << modularity_begin << endl;
#endif

    // Doing Louvain iterations
    int iter_louvain = 0;
    int flag_louvain = 1;
    //while (iter_louvain < 20)
    while(flag_louvain)
    {
        flag_louvain = 0;
#ifdef DOEXP
        iter_louvain++;
        cout << "//////////.......... LOUVAIN ALGO ITER  " << iter_louvain << "  ..........//////////" << endl;
        // PHASE 1 : start to add nodes into groups by consider the " delta_modularity " = ( moving after ) -( moving before )
        cout << endl;
        cout << "     ///// ..... PHASE 1 ..... ///// " << endl; 
        int iter_phase_1 = 0;
#endif
        flag_phase_1 = 1;
        
        //vector<int> node_sequence;
        set<int> node_sequence;
        for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
        {
            //node_sequence.push_back(node_id);
            node_sequence.insert(node_id);
        }
        //cout << node_sequence.size() << endl;



        while(flag_phase_1)
        {
            // For each node start to consider whether it should move to.
#ifdef DOEXP
            iter_phase_1++;
            cout << endl;
            cout << "     // ..... PHASE 1 ITER " << iter_phase_1 << "..... // " << endl << endl;
#endif
            flag_phase_1 = 0;

            // Renew the node vector;
/*            
            set<int> node_sequence;
            for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            {
                node_sequence.insert(node_id);
            }
*/

            // shuffle the vector randomly
            std::random_device rd; // obtain a random seed from the OS
            std::mt19937 gen(rd()); // seed the generator
            //shuffle(begin(node_sequence), end(node_sequence), gen);

            

            //for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            while(!(node_sequence.empty()))
            //for(auto iter = node_sequence.begin() ; iter != node_sequence.end() ; iter++)
            {
                //int node_id = (*iter);
                int node_id = *select_randomly(node_sequence.begin(), node_sequence.end());
                node_sequence.erase(node_id);
                //cout << node_id << endl;
                // Initial for each , clear the neighbor_community
                change_label = graph_info -> label[node_id];
                max_delta_modularity = 0;
                neighbor_community.clear();
                // Find all Communities arround the node and store into neighbor_community "<comm_id, inner_connect between node and comm_id>".
                for(int neighbor_id = 0 ; neighbor_id < graph[node_id].size() ; neighbor_id++)
                {
                    if(neighbor_community.find(graph_info -> label[graph[node_id][neighbor_id]]) != neighbor_community.end())
                    {
                        neighbor_community[graph_info -> label[graph[node_id][neighbor_id]]]++;
                    }
                    else
                    {
                        // pair(community_id, #community_id arround node's neighbor)
                        neighbor_community.insert(pair<int, int>(graph_info -> label[graph[node_id][neighbor_id]], 1));

                    }
                    
                }

                //cout << " Show the neighbor communites arround node " << setw(node_digit) << node_id << endl;

                // For each communities in neighbor_community , start compute delta_modularity , and consider where node should move.
                for(auto map_contain_id = neighbor_community.begin() ; map_contain_id != neighbor_community.end() ; map_contain_id++)
                {
                    //cout << "    comm " << setw(node_digit) << (*map_contain_id).first << " get " << (*map_contain_id).second << " times" << endl;
                    // (*map_contain_id).second :: neighbor community id.
                    // (*map_contain_id).second :: #neighbor community id.
                    delta_modularity = deltaModularity((communities + (*map_contain_id).first), (communities + (graph_info -> label[node_id])), graph[node_id].size(), (*map_contain_id).second, graph_info -> num_edge, graph_info -> loop[node_id]); 
                    //cout << "       delta_modularity = " << delta_modularity << endl;
                    if(delta_modularity > max_delta_modularity)
                    {
                        max_delta_modularity = delta_modularity;
                        change_label = (*map_contain_id).first;
                    }
                }


                // update the node.
                if(change_label != graph_info -> label[node_id])
                {
                    //cout << "    ...Node " << setw(node_digit) << node_id << " move into community " << setw(node_digit) << change_label << " with max_modularity = " << max_delta_modularity << endl;
                    
                    // Moving node out of the original community.
                    if((communities + (graph_info -> label[node_id])) -> member.size() > 1)
                    {
                        //(communities + (graph_info -> label[node_id])) -> inter_connect -= (graph[node_id].size() - neighbor_community[(graph_info -> label[node_id])]);
                        //(communities + (graph_info -> label[node_id])) -> inner_connect -= 2 * neighbor_community[(graph_info -> label[node_id])]; 


                        (communities + (graph_info -> label[node_id])) -> inner_connect -= 2 * neighbor_community[(graph_info -> label[node_id])]; 
                        (communities + (graph_info -> label[node_id])) -> inner_connect -= graph_info -> loop[node_id];
                        (communities + (graph_info -> label[node_id])) -> inter_connect -= graph[node_id].size();
                        (communities + (graph_info -> label[node_id])) -> inter_connect += 2 * neighbor_community[(graph_info -> label[node_id])]; 


                    }
                    else
                    {
                        (communities + (graph_info -> label[node_id])) -> inner_connect -= graph_info -> loop[node_id];
                        (communities + (graph_info -> label[node_id])) -> inter_connect = 0;
                    }
                    (communities + (graph_info -> label[node_id])) -> member.erase(node_id);


                    // Update the infors of moving community.
                    //(communities + change_label) -> inner_connect += (communities + (graph_info -> label[node_id])) -> inner_connect;
                    
                    // Moving node into community with max delta_modularity.
                    graph_info -> label[node_id] = change_label;

                    // Update the infors of moving community.
                    (communities + (graph_info -> label[node_id])) -> member.insert(node_id);
                    //(communities + (graph_info -> label[node_id])) -> inter_connect += (graph[node_id].size() - (2 * neighbor_community[(graph_info -> label[node_id])]));
                    //(communities + (graph_info -> label[node_id])) -> inner_connect += 2 * neighbor_community[(graph_info -> label[node_id])];
                    (communities + (graph_info -> label[node_id])) -> inner_connect += 2 * neighbor_community[(graph_info -> label[node_id])]; 
                    (communities + (graph_info -> label[node_id])) -> inner_connect += graph_info -> loop[node_id];
                    (communities + (graph_info -> label[node_id])) -> inter_connect += graph[node_id].size();
                    (communities + (graph_info -> label[node_id])) -> inter_connect -= 2 * neighbor_community[(graph_info -> label[node_id])]; 
                    flag_phase_1 = 1;
                }

            }
            
        }


        // PHASE 2 : start to merge the gruops , and rebuild the graph
#ifdef DOEXP
        cout << endl;
        cout << "     ///// ..... PHASE 2 ..... ///// " << endl; 
#endif

            int total_connect = 0;



            // Start consider the remain communities.
            num_comm = 0;
            for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            {
                if(!((communities + node_id) -> member.empty()))
                {
                    num_comm++;
                }
            }


            //cout << " REMAIN #COMM = " << num_comm << endl;
            //delete [] final_communities;
            if(final_communities == NULL)
            {
                final_communities = new FinalCommunity[num_comm];
            }
            else
            {
                delete [] final_communities;
                final_communities = new FinalCommunity[num_comm];
            }


            int location = 0;
            for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            {
                if(!((communities + node_id) -> member.empty()))
                {
                    //cout << node_id << endl;
                    for(auto comm_item = (communities + node_id) -> member.begin() ; comm_item != (communities + node_id) ->member.end() ; comm_item++)
                    {
                        //cout << *comm_item <<"  ";
                        //(final_communities + location) -> member.insert(*comm_item);
                        (final_communities + location) -> member.insert((temp_comunities + (*comm_item)) -> member.begin(), (temp_comunities + (*comm_item)) -> member.end());
                        //graph_info -> original_label[*comm_item] = location;
                        graph_info -> label[*comm_item] = location;
                    }
                    (final_communities + location) -> inner_connect += (communities + node_id) -> inner_connect;
                    (final_communities + location) -> inter_connect += (communities + node_id) -> inter_connect;
                    location++;
                }
            }


            // compress the communities.
           
            delete [] communities;
            Community* temp_1 = new Community [num_comm];
            communities = temp_1;
            //cout << &communities << endl;
            Community* *temp = &communities;

            for(int comm_id = 0 ; comm_id < num_comm ; comm_id++)
            {
                (communities + comm_id) -> member.insert(comm_id);
            }




            for(int comm_id = 0 ; comm_id < num_comm; comm_id++)
            {
                for(auto comm_item = (final_communities + comm_id) -> member.begin() ; comm_item != (final_communities + comm_id) ->member.end() ; comm_item++)
                {
                    graph_info -> original_label[(*comm_item)] = comm_id;
                    //graph_info -> label[*comm_item] = comm_id;
                }
            }

            if(graph_info -> end_node == graph_info -> original_end_node)
            {
                for(int comm_id = 0 ; comm_id < num_comm; comm_id++)
                {
                    for(auto comm_item = (final_communities + comm_id) -> member.begin() ; comm_item != (final_communities + comm_id) ->member.end() ; comm_item++)
                    {
                        //graph_info -> original_label[(*comm_item)] = comm_id;
                        graph_info -> label[*comm_item] = comm_id;
                    }
                }
            }

 
            delete [] temp_2;
            FinalCommunity* temp_2 = new FinalCommunity[num_comm];
            temp_comunities = temp_2;
            //FinalCommunity* *tamp = &temp_comunities;


            for(int i = 0 ; i < num_comm ; i++)
            {
                *(temp_comunities + i) = *(final_communities + i);
            }


            // Rebuild 
                // Using a temp to store the rebuild graph.

            free(graph_info -> degree);
            graph_info -> degree = (int*)calloc(num_comm, sizeof(int));
            vector<vector<int>> graph_temp(num_comm);



                // Build the graph and update the compress comm's infos.
            //cout << " Rebuild the graph ......................" << endl;
            for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            {
                for(int edge_id = 0 ; edge_id < graph[node_id].size() ; edge_id++)
                {
                    if(graph_info -> label[node_id] != graph_info -> label[graph[node_id][edge_id]])  // Both twos node are'nt in the same community , than add an edge between the compress nodes.
                    {
                        graph_temp[graph_info -> label[node_id]].push_back(graph_info -> label[graph[node_id][edge_id]]);
                        graph_info -> degree[graph_info -> label[node_id]]++;
                    }
                    // else // Both two nodes are in the same community , than compress into one node and add a loop.
                }
            }
                // Update the graph infos.
            graph_info -> start_node = 0;
            graph_info -> end_node = num_comm - 1;

            //cout << "  ----- label after rebuild the graph -----" << endl;
            for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
            {
                graph_info -> label[node_id] = node_id;
                graph_info -> loop[node_id] = (final_communities + node_id) -> inner_connect;
                //cout << " label " << node_id << " : " << graph_info -> label[node_id] << endl;
                (communities + node_id) -> inner_connect = (final_communities + node_id) -> inner_connect;
                (communities + node_id) -> inter_connect = (final_communities + node_id) -> inter_connect;
            }


            // Assign the rebuild graph to the original graph.
            graph.clear();
            graph = graph_temp;


            for(int i = 0 ; i < num_comm ; i++)
            {
                total_connect += (communities + i) -> inner_connect;
                total_connect += (communities + i) -> inter_connect;
            }

            //*(mod_record + iter_louvain) = modularityCompute(communities, graph_info -> num_edge, graph_info -> start_node, graph_info -> end_node);
            modularity = modularityCompute(communities, graph_info -> num_edge, graph_info -> start_node, graph_info -> end_node);
            mod_record.push_back(modularity);
            if((max_modularity - modularity))
            {
                max_modularity = modularity;
                flag_louvain = 1;
            }
    }


#ifdef DOEXP

    cout << " MOULARITY :: " << endl;
    cout << " before doing louvain algo = " <<  modularity_begin << endl;

    for(int record_id = 0 ; record_id < mod_record.size() ; record_id++)
    {
        cout << "   MOD IN  " << record_id  << "  iter = " << mod_record[record_id] << endl;
    }
    cout << "NUM OF COMM = " << num_comm << endl;
    cout << "NUM OF ITERATION = " << iter_louvain << endl;
#endif

    return max_modularity;

/*
    // SHOW FINAL COMMUNITITES.
    int total_connect = 0;
    cout << " ..... FINAL COMMUNITY ....." << endl;
    for(int i = 0 ; i < num_comm ; i++)
    {
        cout << "COMM " << i << " CONTAIN :: " << endl;
        for(auto iter = (final_communities + i) -> member.begin() ; iter != (final_communities + i) -> member.end() ; iter++)
        {
            cout << "   " << *iter;
        }
        cout << endl;
        cout << "     inner connect = " << (final_communities + i) -> inner_connect << endl;
        cout << "     inter connect = " << (final_communities + i) -> inter_connect << endl;
        total_connect += (final_communities + i) -> inner_connect;
        total_connect += (final_communities + i) -> inter_connect;
    }
    cout << "/////................................. total connect ===> " << total_connect << endl;
    cout << endl;
            


    // SHOW ORIGINAL LABEL.
    cout << " ..... ORIGINAL LABEL ....." << endl;
    for(int node_id = graph_info -> original_start_node ; node_id <= graph_info -> original_end_node ; node_id++)
    {
        cout << " label " << node_id << " : " << graph_info -> original_label[node_id] << endl;
    }
*/


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
        graph_info -> start_node  = 0;
        graph_info -> original_start_node = read_node1;
        graph_info -> end_node = graph_info -> num_node - 1;             // MUST MODIFY !!! CAUSE THE NODE START IN 0 AND END IN NUM_NODE-1 !!!
        graph_info -> original_end_node = graph_info -> num_node - 1;
    }
    else
    {
        graph_info -> start_node = 1;
        graph_info -> original_start_node = read_node1;
        graph_info -> end_node = graph_info -> num_node; 
        graph_info -> original_end_node = graph_info -> num_node;
    }
    // Create a space to store graph and it's degree.
    vector<vector<int>> graph(graph_info -> num_node + 1);
    graph_info -> degree = (int*)calloc((graph_info -> num_node + 1), sizeof(int));
    graph_info -> original_degree = (int*)calloc((graph_info -> num_node + 1), sizeof(int));


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
        graph_info -> original_degree[read_node1]++;
        graph[read_node2].push_back(read_node1);
        graph_info -> degree[read_node2]++;
        graph_info -> original_degree[read_node2]++;
    }
    
    input_file.close();
    return(graph);
}

int main(int argc, char* argv[], char* env[])
{// ./Louvain.exe  SRCFile LOGFile GTFile EXPFile[enter]
    // SRCFile :: dataset/dataset.name.txt
    // LOGFile ::dataset_log/dataset.name_log.txt
    // GTFile :: dataset_gt/dataset.name.GT.txt
    // EXPFile :: exp_record/dataset.name_exp_record.txt

    GraphInfo* graph_info = (GraphInfo*)malloc(1 * (sizeof(GraphInfo)));
    vector<vector<int>> graph;


    // Recording execution time
    clock_t start, end;
    double read_build_graph_time_record;
    double algo_time_record;

    float modularity = 0.0;
    
    if(argc == 5)
    {
        clock_t start, end;

        start = clock();
        graph = (vector<vector<int>>) readFile(argv[1], graph_info);
        end = clock();

    }
    else
    {
        cout << "USAGE: ./Louvain.exe   SRCFile LOGFile GTFile EXPFile[enter]" << endl;
        exit(1);
    }


    start = clock();

    modularity = Louvain(graph_info, graph);

    end = clock();

    algo_time_record = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "          ///// ..... EXECUTION TIME ..... /////          "  << endl;
    cout << " READ BUILD GRAPH TIME TAKES :: " << fixed << read_build_graph_time_record << " sec " << endl;
    cout << " ALGO LOUVAIN TAKES :: " << fixed << algo_time_record << " sec " << endl;



        // Output the results.
    // Writing the label record to file [dataset.name]_log.txt.
    cout << "WRITTING THE RESULT INTO LOG FILE ." << endl;
    writeFile(argv[2], graph_info);
    cout << "FINISH WRITTING !!! " << endl;





    // Caculating the measurements.
    cout << "///////////////..... EXPERIMENT MEASUREMENTS ..... ////////////////" << endl << endl;


    map<int,int> label_count;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        label_count[graph_info -> label[node_id]]++;
    }

    int num_comm = label_count.size();
    cout << " TOTAL #COMMUNITY =  " << num_comm << endl; 

    // Compute NMI.
     Dataset_info* dataset = new Dataset_info[sizeof(Dataset_info)];

        // Read SRCFile and GTFile.
    cout << " Reading GT dataset ... " << endl;
    readGT(argv[2], argv[3], dataset);

        // Partition the dataset into groups.
     cout << " Partitioning dataset ... " << endl;
    partitionDataset(dataset);

    vector<set<int>> algo_partition(dataset -> algo_num_comm);
    vector<set<int>> gt_partition(dataset -> gt_numn_comm);

    cout << " Building algo_partition and gt_partition ... " << endl;
    buildPartition(dataset, algo_partition, gt_partition);

        // Compute NMI.
    cout << " Computing NMI ... " << endl;
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
    cout << "    READ BUILD GRAPH TIME = " << read_build_graph_time_record << "sec"  << endl;
    cout << "    ALGO TIME             = " << algo_time_record << "sec" << endl;
    cout << "    NUM COMMUNITY         = " << num_comm << endl;
    cout << "    MODULARITY            = " << modularity << endl;
    cout << "    NMI                   = " << NMI << endl;
    cout << endl;

    return 0;






    

/*
    cout << "SHOW THE LABEL OF EACH NODE :: " << endl;
    for(int node_id = graph_info -> original_start_node ; node_id <= graph_info -> original_end_node ; node_id++)
    {
        cout << "NODE " << setw(node_digit) << node_id << " : " << setw(node_digit) << graph_info -> original_label[node_id] << endl;
    }
    //cout << "hahaha" << endl;
*/
}






//............................................................................. TRACING FUNCTION
/*

    // Show communities infos ::

    total_connect = 0;
    cout << " COMMUNITIES .............." << endl;
    for(int i = 0 ; i < num_comm ; i++)
    {
        cout << "COMM " << i << " CONTAIN :: " << endl;
        for(auto iter = (communities + i) -> member.begin() ; iter != (communities + i) -> member.end() ; iter++)
        {
            cout << "   " << *iter;
        }
        cout << endl;
        cout << "     inner connect = " << (communities + i) -> inner_connect << endl;
        cout << "     inter connect = " << (communities + i) -> inter_connect << endl; 
        total_connect += (communities + i) -> inner_connect;
        total_connect += (communities + i) -> inter_connect;
    }
    cout << "/////................................. total connect ===> " << total_connect << endl;
    cout << endl;





    // show the graph ::

    cout << " GRAPH .............." << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        out << "NODE " << setw(node_digit) << node_id << endl;
        for(int edge_id = 0 ; edge_id < graph[node_id].size() ; edge_id++)
        {
            cout << "  " << setw(node_digit) << graph[node_id][edge_id] << " ";
        } 
        cout << endl;
        cout << " degree = " << graph_info -> degree[node_id] << endl;
    }




    // Show neighbor communities ::

    for(auto map_contain_id = neighbor_community.begin() ; map_contain_id != neighbor_community.end() ; map_contain_id++)
    {
        cout << "    comm " << setw(node_digit) << (*map_contain_id).first << " get " << (*map_contain_id).second << " times" << endl;
        // (*map_contain_id).second :: neighbor community id.
        // (*map_contain_id).second :: #neighbor community id.
    }



    // Show all community members :: 

    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << ".......... Community " << setw(node_digit) << node_id << " contain members .......... " << endl;
        for(auto member_id = (communities + node_id) -> member.begin() ; member_id != (communities + node_id) -> member.end() ; member_id++)
        {
            //cout << *node_id << " ";
            cout << setw(node_digit) << *member_id << " ";
        }
        cout << endl;
    }



    // Show all labels ::

    cout << "SHOW THE LABEL OF EACH NODE :: " << endl;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        cout << "NODE " << setw(node_digit) << node_id << " : " << setw(node_digit) << graph_info -> label[node_id] << endl;
    }

*/