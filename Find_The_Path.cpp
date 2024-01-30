/** BFS based shortest path finder algo.
* finds shortest path starting from 'S' to 'E'
* I am thinking of extending it to a graphical 2D game with sockets
*/
#include<bits/stdc++.h>
using namespace std;
#define X first
#define Y second
int dir1[]={1,0,-1,0};
int dir2[]={0,1,0,-1};
/*
4 5
E..##
##..#
##..#
##.S#
*/
string getMove(int d){
    string move;
    switch(d){
        case 0:
            move="Down";
            break;
        case 1:
            move="Right";
            break;
        case 2:
            move="Up";
            break;
        case 3:
            move="Left";
            break;
        default:
            move="We are lost in the grid!";
    }
    return move;
}
int main(){
    int n,m;
    cin>>n>>m;
    char grid[n][m];
    pair<int,int>start,end;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            cin>>grid[i][j];
            if(grid[i][j]=='S'){
                //this is start point
                start={i,j};
            }
            else if(grid[i][j]=='E'){
                //this is end point
                end={i,j};
            }
        }
    }
    int dist[n][m];
    memset(dist,-1,sizeof(dist));
    dist[end.X][end.Y]=0;
    deque<pair<int,int>>queue;
    queue.push_back(end);
    while(!queue.empty()){
        auto cur_cell=queue.front();
        //cout<<cur_cell.X<<" "<<cur_cell.Y<<" :"<<dist[cur_cell.X][cur_cell.Y]<<endl;
        queue.pop_front();
        for(int d=0;d<4;d++){
            int next_x=cur_cell.X+dir1[d],next_y=cur_cell.Y+dir2[d];
            if(next_x<0||next_x>=n||next_y<0||next_y>=m)continue;//invalid coord..
            if(grid[next_x][next_y]!='#'&&~dist[next_x][next_y]){
                dist[next_x][next_y]=dist[cur_cell.X][cur_cell.Y]+1;
                queue.push_back({next_x,next_y});
            }
        }
    }
    if(~dist[start.X][start.Y]){
        cout<<"a possible Shortest path in "<<dist[start.X][start.Y]<<" steps;\nHere are the steps:"<<endl;
        //complete here
        pair<int,int>cur=start;
        while(grid[cur.X][cur.Y]!='E'){
            for(int d=0;d<4;d++){
                int next_x=cur.X+dir1[d],next_y=cur.Y+dir2[d];
                if(next_x<0||next_x>=n||next_y<0||next_y>=m)continue;//invalid coord..
                if(grid[next_x][next_y]=='#')continue;
                if(dist[next_x][next_y]==dist[cur.X][cur.Y]-1){
                    //possible path
                    cur={next_x,next_y};
                    cout<<getMove(d)<<endl;
                    break;
                }
            }
        }
    }
    else{
        cout<<"No path found!"<<endl;
    }
}
