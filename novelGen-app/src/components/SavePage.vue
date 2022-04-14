<template>
  
  <el-table :data="tableData.filter(data => !search || data.txtTitle.toLowerCase().includes(search.toLowerCase()))" :key="Math.random()"
     style="width: 100%" v-show="tablevisible">
     <el-table-column label="时间">
      <template #default="scope" sortable>
        <div style="display: flex; align-items: center">
          <el-icon><clock /></el-icon>
          <span style="margin-left: 10px">{{ scope.row.date }}</span>
        </div>
      </template>
    </el-table-column>
    <!-- <el-table-column label="时间" prop="date" sortable/> -->
    <el-table-column label="标题" prop="txtTitle" sortable/>
    <el-table-column align="right">
      <!-- <el-icon><clock /></el-icon> -->
      <template #header>
        <el-input v-model="search" size="small" placeholder="关键词搜索" />
      </template>
      <template #default="scope">
        <!-- <el-button
          size="mini"
          @click="handleEdit(scope.$index, scope.row)">Edit</el-button> -->
        <el-button color="#a899ef" plain style="margin-left: 16px" @click="handleOpen(scope.row)">查看</el-button>
        <el-popconfirm title="确认删除此作品吗" @confirm="handleDelete(scope.row)" icon-color="red">
          <template #reference>
            <el-button type="danger" style="margin-left: 16px">删除</el-button>
          </template>
        </el-popconfirm>
      </template>
    </el-table-column>
  </el-table>
  <el-drawer v-model="drawer" :title="showText.txtTi" direction="rtl" size="50%">
    <span style="white-space: pre-line">{{showText.txtTe}}</span>
  </el-drawer>
  <el-container class="showButtonCon" v-show="getbutton">
    <el-button class="showTextButton"  @click="getText" size="large" color="#a899ef" plain>展示作品集</el-button>
  </el-container>
</template>

<script>

export default {
  data(){
    return{
      drawer:false,
      tablevisible:false,
      getbutton:true,
      search:'',
      tableData:[],
      showText:{
        txtTi:'我是标题',
        txtTe:'我是文本'
      }
    }
  },
  methods: {
      handleOpen(row) {
        this.drawer=true;
        //row.txtText
        this.showText.txtTi = row.txtTitle;
        this.showText.txtTe = row.txtText
        /* var str = row.txtText;
        var reg=new RegExp("\n","g");
        str = str.replace(reg,"<br>");
        this.showText.txtTe = str; */
      },
      handleDelete(row){
        const self = this;
        let params=self.$qs.stringify({name:self.$store.state.user.username,title:row.txtTitle})
        self.$http({
            method:'post',
            url:'http://localhost:5000/api/deleteText',
            data:params
        })
        .then(res =>{
            /* console.log('res=>',res);*/
            /* console.log('res=>',res.data);  */
            if(res.data==1){
              this.$notify({
                  type:'success',
                  message:'成功删除《'+row.txtTitle+'》!',
                  duration:3000
              })
              this.getText()
            }
            else{
              this.$notify({
                  type:'error',
                  message:'删除《'+row.txtTitle+'》失败,请重试!',
                  duration:3000
              })
            }
        } )
        .catch( err =>{
            console.log(err);
        })
      },
      getText(){
        const self = this;
        let params=self.$qs.stringify({name:self.$store.state.user.username})
        self.$http({
            method:'post',
            url:'http://localhost:5000/api/getText',
            data:params
        })
        .then(res =>{
            /* console.log('res=>',res);*/
            /* console.log('res=>',res.data);  */
            this.tableData = res.data;
            this.tablevisible=true;
            this.getbutton=false;
        } )
        .catch( err =>{
            console.log(err);
        })
      }
    },
}
</script>

<style>
/* .showTextButton{
  #text-align:center;
  
} */
.showButtonCon{
  position:absolute;
  top:50%;
  left: 50%;
  transform: translate(-50%,-50%);
}
</style>