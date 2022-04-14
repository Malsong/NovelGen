<template>
  <el-row class="editpage"> <!-- editpage：整个大的生成页 -->
    <el-col :span="15" :xs="24" :sm="15" :md="15" :lg="15" :xl="15">
        <div class="edit"><!-- edit：整个左侧编辑栏 -->
            <div class="editAllBox"><!-- editAllBox：整个左侧输入输出框 -->
                <el-row class="title">
                    <el-input size="large" type="text" autocomplete="off" placeholder="无标题"  clearable  v-model="txtTitle"/>
                </el-row>
                <el-row class="textarea">
                    <div class="edit-write">
                        <!-- <div id="editor" class="txt" type="textarea" placeholder="请输入小说前情提要" contenteditable="true" spellcheck="false" autofocus="autofocus">
                        
                        </div> -->
                        
                        <el-input type="textarea" :rows="18" class="inAndOut" v-model="nowtext">{{chooseTxt}}</el-input>
                        
                        <!-- <textarea ></textarea> -->
                    </div>
                </el-row>
                 <el-row class='saveC'>
                        <el-button-group>
                            <el-button color="#a899ef" plain @click="getText" v-if="$store.state.stateOn">
                                <el-icon><upload /></el-icon><span>导入</span>
                            </el-button>
                            <el-button color="#a899ef" plain @click="saveText" v-if="$store.state.stateOn">
                                <el-icon><document-checked /></el-icon><span>存储</span></el-button>
                            <el-button color="#a899ef" plain @click="genText"><el-icon><edit-pen /></el-icon><span>继续生成</span></el-button>
                        </el-button-group>
                    <!-- <el-col :span='12'>继续生成</el-col> -->
                </el-row>
                
            </div>
            <el-dialog title="小说作品集" v-model="dialogTableVisible" :modal-append-to-body="false" width="70%">
                <el-table :data="gridData" ref="singleTable" highlight-current-row @current-change="handleCurrentChange" height="300">
                <el-table-column type="index" width="50"></el-table-column>
                <el-table-column property="date" label="时间" ></el-table-column>
                <el-table-column property="txtTitle" label="标题" ></el-table-column>
                <el-table-column property="txtText" label="文本" ></el-table-column>
                </el-table>
                <template #footer>
                <span>
                    <el-button @click="cancelCurrent()">取消</el-button>
                    <el-button type='primary' @click="addToEdit" >确定</el-button>
                </span>
                </template>
            </el-dialog>
            <!-- @current-change="handleCurrentChange" -->
            <el-dialog title="提示" width="300px" v-model="dialogTableVisible2" :modal-append-to-body="false">
                <span>已有同名小说，确认覆盖原文本吗？</span>
                <template #footer>
                <span class="dialog-footer">
                    <el-button @click="dialogTableVisible2=false">取消</el-button>
                    <el-button type="primary" @click="coverText" >确定</el-button>
                </span>
                </template>
            </el-dialog>
        </div>
        <!-- <div class="grid-content bg-purple"></div> -->
    </el-col>
    <el-col :span="9" :xs="0" :sm="9" :md="9" :lg="9" :xl="9" >
        <div class="model">
            <div class="modelBox">
                
                <el-row class="typeChoose" justify="center">
                    <!-- <div>创作方式：</div> -->
                    <el-radio-group v-model="isFreedom" size="large" fill="#d1c8fd" >
                        <el-radio-button label='1'>自由创作</el-radio-button>
                        <el-radio-button label='2'>著作续写</el-radio-button>
                    </el-radio-group>
                       
                </el-row >
                <el-row class="typeBox1" v-show="isFreedom=='1'" justify="center">
                    <!-- <div>创作类型：</div> -->
                    <el-select class="choose1" placeholder="请选择小说类型" v-model="genCondition.m_class">
                        <el-option v-for="item in options1" :key="item.value" :label="item.label" :value="item.value" />
                    </el-select>
                    <el-button color="#a899ef" plain @click="genText">生成</el-button>
                </el-row>
                <el-row class="typeBox2" v-show="isFreedom=='2'" justify="center">
                    <el-select class="choose2" placeholder="请选择小说类型" v-model="genCondition.m_class">
                        <el-option v-for="item in options2" :key="item.value" :label="item.label" :value="item.value" />
                    </el-select>
                    <el-button color="#a899ef" plain @click="genText">生成</el-button>
                </el-row>
                
            </div>
            
        </div>
        <div class="eg">
            <div class="egBox">
                <div class="allBox" v-loading="loading">
                    <el-col :span="3" class="radioBox">
                        <el-radio-group class="radioGroup" size="large" v-model="txtradio" >
                            <el-row class="egradio"><el-radio  :label="1">&nbsp;</el-radio></el-row>
                            <el-row class="egradio"><el-radio  :label="2">&nbsp;</el-radio></el-row>
                            <el-row class="egradio"><el-radio  :label="3">&nbsp;</el-radio></el-row>
                        </el-radio-group>
                    </el-col>
                    <el-col :span="21" class="txtBox">
                        <el-row v-for="(v,k) in textList" class="oneBox"><p class="onBox_show">{{v}}</p></el-row>
                    </el-col>
                    
                </div>
            </div>
        </div>
        <div class="addAndChange">
            <el-tooltip placement="top" effect="customized">
                <template #content>① 上方圆形按键选择文本<br />② 此键将选中文本添加到左方文本框中 </template>
            <el-button class="add" color="#a899ef" plain size='small' @click="chooseRadio">+</el-button>
            </el-tooltip>
            <div class="changeTip" v-if="isGen">
                <span>没有满意的？</span>
                <el-button color="#a899ef" plain @click="genText">换一批</el-button>
            </div>
        </div>
        <!-- <div class="grid-content bg-purple-light"></div> -->
    </el-col>
  </el-row>
  
</template>

<script>
export default {
    data(){
        return {
            txtTitle:'',
            isGen:false,
            isFreedom:'1',
            txtradio:'',
            nowtext:'',
            loading:false,
            options1:[
                {
                    value:'A',
                    label:'悬疑灵异'
                },
                {
                    value:'B',
                    label:'青春校园'
                },
                {
                    value:'C',
                    label:'古言架空'
                }
            ],
            options2:[
                {
                    value:'E',
                    label:'《甄嬛传》'
                },
                {
                    value:'F',
                    label:'《法医秦明》'
                },
                {
                    value:'G',
                    label:'《鬼吹灯》'
                }
            ],
            genCondition:{
                prefix:'',
                length:100,
                m_class:'',
                temperature:0.8
            },
            textList:{
                text1:'',
                text2:'',
                text3:''
            },
            dialogTableVisible:false,
            gridData:[],
            originalText:null,
            dialogTableVisible2:false
        }
    },
    methods: {
        genText(){
            this.txtradio = '';
            const self = this;
            if (self.genCondition.m_class!=''){
                this.loading = true;
                let params = self.$qs.stringify({
                    prefix:self.nowtext,
                    length:self.genCondition.length,
                    m_class:self.genCondition.m_class,
                    temperature:self.genCondition.temperature
                })
                self.$http({
                    method:'post',
                    url:'http://localhost:5000/api/generateText',
                    data:params
                })
                .then(res => {
                    this.textList.text1 = res.data[0]
                    this.textList.text2 = res.data[1]
                    this.textList.text3 = res.data[2]
                    this.loading = false;
                })
                this.isGen = true;
            }
            else{
                this.loading = false;
                this.$notify({
                    type:'error',
                    message:'请选择小说类别!',
                    duration:5000
                })
            }
        },
        saveText(){
            const self = this;
            //获取当前用户小说标题
            if(self.txtTitle!=''){
                let params1=self.$qs.stringify({name:self.$store.state.user.username,title:self.txtTitle})
                self.$http({
                    method:'post',
                    url:'http://localhost:5000/api/haveSame',
                    data:params1
                })
                .then(res =>{
                    console.log('res=>',res.data);  
                    if(res.data==2){
                        //无重名
                        var dateNow = new Date();
                        let params = self.$qs.stringify({
                                name:self.$store.state.user.username,
                                date:dateNow.getFullYear()+'-'+dateNow.getMonth()+'-'+dateNow.getDay()+' '+dateNow.getHours()+':'+dateNow.getMinutes()+':'+dateNow.getSeconds(),
                                title:self.txtTitle,
                                text:self.nowtext
                            })
                            self.$http({
                                method:'post',
                                url:'http://localhost:5000/api/saveText',
                                data:params
                            })
                            .then(res => {
                                switch(res.data){
                                    case 0:
                                        this.$notify({
                                            type:'error',
                                            message:'系统繁忙，请稍后再试！',
                                            duration:3000
                                        })
                                        break;
                                    case 1:
                                        this.$notify({
                                            type:'success',
                                            message:'小说：《'+self.txtTitle+'》保存成功！',
                                            duration:3000
                                        })
                                        break;
                                }
                            })
                            .catch( err =>{
                                console.log(err);
                            })
                    }else if(res.data==1){
                        this.dialogTableVisible2 = true;
                    }else{
                        this.$notify({
                            type:'error',
                            message:'系统繁忙，请稍后再试！',
                            duration:3000
                        })
                    }
                    /* console.log(allData);   */
                    /* for(var i =0 ; i<allData.length;i++){
                        if (allData[i]['txtTitle']==self.txtTitle){
                            existed = true;
                        }else{
                            existed = false;
                        }
                    } */
                } )
                .catch( err =>{
                    console.log(err);
                })
            }else{
                this.$notify({
                    type:'error',
                    message:'小说标题不能为空！',
                    duration:3000
                })
            }
            
            /* const self2 = this;
            for(var i =0 ; i<this.gridData.length;i++){
                if (this.gridData[i]['txtTitle']==self.txtTitle){
                    existed = true;
                }else{
                    existed = false;
                }
            }
            console.log(existed); */
            /* console.log(existed) */
            /* if(existed==false){
                if(self.txtTitle!=''){
                    var dateNow = new Date();
                    let params = self.$qs.stringify({
                            name:self.$store.state.user.username,
                            date:dateNow.getFullYear()+'-'+dateNow.getMonth()+'-'+dateNow.getDay()+' '+dateNow.getHours()+':'+dateNow.getMinutes()+':'+dateNow.getSeconds(),
                            title:self.txtTitle,
                            text:self.nowtext
                        })
                        self.$http({
                            method:'post',
                            url:'http://localhost:5000/api/saveText',
                            data:params
                        })
                        .then(res => {
                            switch(res.data){
                                case 0:
                                    this.$notify({
                                        type:'error',
                                        message:'系统繁忙，请稍后再试！',
                                        duration:3000
                                    })
                                    break;
                                case 1:
                                    this.$notify({
                                        type:'success',
                                        message:'小说：《'+self.txtTitle+'》保存成功！',
                                        duration:3000
                                    })
                                    break;
                            }
                        })
                        .catch( err =>{
                            console.log(err);
                        })
                }else{
                    this.$notify({
                        type:'error',
                        message:'小说标题不能为空！',
                        duration:3000
                    })
                }
            }else{ //存在同标题文本
                this.dialogTableVisible2=true
                console.log(this.dialogTableVisible2)
            } */
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
                /* console.log('res=>',res);
                console.log('res=>',res.data);  */
                this.gridData = res.data;
                this.dialogTableVisible = true
            } )
            .catch( err =>{
                console.log(err);
            })
        },
        coverText(){
            const self = this;
            var dateNow = new Date();
            let params=self.$qs.stringify({
                name:self.$store.state.user.username,
                date:dateNow.getFullYear()+'-'+dateNow.getMonth()+'-'+dateNow.getDay()+' '+dateNow.getHours()+':'+dateNow.getMinutes()+':'+dateNow.getSeconds(),
                title:self.txtTitle,
                text:self.nowtext
            })
            self.$http({
                method:'post',
                url:'http://localhost:5000/api/coverText',
                data:params
            })
            .then(res =>{
                /* console.log('res=>',res);
                console.log('res=>',res.data);  */
                switch(res.data){
                    case 0:
                        this.$notify({
                            type:'error',
                            message:'系统繁忙，请稍后再试！',
                            duration:3000
                        })
                        this.dialogTableVisible2 = false;
                        break;
                    case 1:
                        this.$notify({
                            type:'success',
                            message:'小说《'+self.txtTitle+'》修改成功！',
                            duration:3000
                        })
                        this.dialogTableVisible2 = false;
                        break;
                }
            } )
            .catch( err =>{
                console.log(err);
            })
        },
        cancelCurrent(row){
            /* console.log(this.originalText) */
            this.$refs.singleTable.setCurrentRow(row);
            this.dialogTableVisible = false;
            /* console.log(this.originalText) null */ 
        },
        handleCurrentChange(val){
            this.originalText=val;
        },
        addToEdit(){
            /* console.log(this.originalText) */
            const self = this;
            if(self.originalText!=null){
                this.txtTitle=self.originalText.txtTitle;
                this.nowtext=self.originalText.txtText;
                this.dialogTableVisible = false;
            }else{
                this.$notify({
                    type:'warning',
                    message:'请确认是否选中！',
                    duration:3000
                })
            }
        },
        chooseRadio(){
            if (this.txtradio=='1'){
                this.nowtext = this.nowtext+this.textList.text1;
            }else if(this.txtradio=='2'){
                this.nowtext = this.nowtext+this.textList.text2;
            }else if(this.txtradio=='3'){
                this.nowtext = this.nowtext+this.textList.text3;
            }
            /* console.log(this.nowtext) */
        }
        /* chooseTxt(){
             if (this.txtradio=='1'){
                this.nowtext = this.nowtext+this.textList.text1;
            }else if(this.txtradio=='2'){
                this.nowtext = this.nowtext+this.textList.text2;
            }else if(this.txtradio=='3'){
                this.nowtext = this.nowtext+this.textList.text3;
            }else{
                this.nowtext = this.nowtext;
            }
        } */
    },
    computed:{
        chooseTxt:function(){
            this.nowtext = this.nowtext
            /* if (this.txtradio=='1'){
                this.prefix = this.nowtext+this.textList.text1;
            }else if(this.txtradio=='2'){
                this.prefix = this.nowtext+this.textList.text2;
            }else if(this.txtradio=='3'){
                this.prefix = this.nowtext+this.textList.text3;
            }else{
                this.prefix = this.nowtext;
            } */
        }
    }
}
</script>

<style>
.edit{
    padding-left: 20px;
    padding-right: 20px;
    flex: 2;
    padding-top: 20px;
    min-width: 0;
    position: relative;
    transition: paddingTop .4s;
    height: 85vh;
}
.editAllBox{
    /* padding-left: 0;
    padding-right: 10px; */
    height: 100%;
    width: 100%;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12),0 0 6px rgba(0,0,0,0.04);
    border-radius:10px;
}
.title{
    border: none;
    padding-left: 5%;
    padding-right: 5%;
    padding-top: 2%;
    padding-bottom: 2%;
}
.textarea{
    width:100%;
    height: 73%;
    padding-left: 5%;
    padding-right: 5%;
    /* padding-bottom: 65vh; */
}
.edit-write{
    width:100%;
    height: 100%;
}
.inAndOut{
    width:100%;
    height:100%;
}
.scrollbar-demo-item {
  /* display: flex;
  align-items: center;
  justify-content: center; */
  height: 30px;
  margin: 5px;
  /* text-align: center; */
  border-radius: 4px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
}
.el-input_inner{
    border: none;
    padding: 0;
    font-size: 30px;
    font-weight: 600;
    color: #1f1f1d;
    /* background-color: initial; */
    /* -webkit-appearance: none; */
    /* box-sizing: border-box;
    display: inline-block; */
    /* height: 50px;
    line-height: 40px; */
    /* outline: 0;
    transition: border-color .2s cubic-bezier(.645,.045,.355,1); */
    /* width: 100%; */
}
.saveC{
    padding-left: 5%;
}
.model{
    /* background: #fbfaf9; */
    padding-top: 20px;
    padding-bottom: 20px;
    padding-right: 20px;
    padding-left: 20px;
    height: 25vh;
    position: relative;
    flex: 1;
    min-width: 0;
    transition: background-color .05s;
    display: flex;

}
.modelBox{
    padding-right: 10px;
    padding-left: 10px;
    height: 100%;
    width:100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    /* padding-top: 30px; */
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12),0 0 6px rgba(0,0,0,0.04);
    border-radius:10px;
}
.typeBox{
    padding-top: 1%;
    padding-left: 1%;
    padding-right: 1%;
}  
.typeChoose{
    padding-top: 5%;
}
.txtButton{
    flex-direction: column;
}
.eg{
    height: 50vh;
    /* padding-top: 20px;
    padding-bottom: 20px; */
    padding-right: 20px;
    padding-left: 20px;
    position: relative;
    flex: 1;
    min-width: 0;
    transition: background-color .05s;
    display: flex;
}
.egBox{
    padding-right: 2%;
    padding-left: 2%;
    height: 100%;
    width:100%;
    /* flex: 1; */
    display: flex;
    flex-direction: column;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12),0 0 6px rgba(0,0,0,0.04);
    border-radius:10px;
}
.allBox{
    padding-top: 2%;
    padding-bottom: 2%;
    padding-left: 1%;
    padding-right: 1%;
    height: 96%;
    width:98%;
}
.radioBox{
    height: 100%;
    width:100%;
}
.radioGroup{
    height: 100%;
    width:100%;
}
.egradio{
    padding-left: 30%;
    text-align: center;
}
.txtBox{
    height: 100%;
    width:100%;
}
.oneBox{
    padding-left: 2%;
    padding-right: 2%;
    /* padding-bottom: 2%;
    padding-top: 2%; */
    height: 28%;
    width:90%;
    position: flex;
    border-color: #d1c8fd;
    border-width:1px;
    border-style:solid;
    border-radius:10px;
}
.onBox_show{
    padding-left: 2%;
    padding-right: 2%;
    padding-top: 2%;
    width:96%;
    height: 80%;
    font-family:"宋体";
    font-size:14px;
    color: #a899ef;
    word-wrap: break-word;
    word-break: break-all;
    overflow:hidden;
    overflow-y:scroll;
}
.onBox_show::-webkit-scrollbar{
    display:none;/*隐藏滚动条*/
}
.addAndChange{
    /* height: 50vh; */
    /* padding-inline-end: 5%;
    padding-inline-start: 5%; */
    padding-top: 2%;
    padding-left:5% ;
    width: 95%;
    
}
.changeTip{
    
    /* padding-bottom: 2%; */
    text-align: center;
    position:relative;
    z-index: 5;
}
.add{
    padding-top: 2%;
    padding-left: 2%;
    position:absolute;
    z-index: 10;
}
.txt{
    width: 100%;
    box-sizing: border-box;
    padding-bottom: 15px;
    word-wrap: break-word;
    word-break: break-word;
    line-height: 26px;
    color: #1f1f1d;
    cursor: text;
    z-index: 10;
    position: relative;
    font-weight: 400;
    white-space: pre-wrap;
    -webkit-user-modify: read-write;
    /* -webkit-line-break: after-white-space; 有黄线*/
}
.el-text{
    padding-left: 0;
    padding-right: 10px;
    position: relative;
}
.editpage{
    height:90vh;
    background: url('../pic/genbg.png');
    z-index: 200px;
}


.el-radio__input.is-checked .el-radio__inner {
  background: #a899ef !important;
  border-color: #d1c8fd !important;
}

.el-row {
  margin-bottom: 20px;
}
.el-row:last-child {
  margin-bottom: 0;
}
.el-col {
  border-radius: 4px;
}
.bg-purple {
  background: #d3dce6;
}
.bg-purple-light {
  background: #e5e9f2;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.row-bg {
  padding: 10px 0;
  background-color: #f9fafc;
}
.el-popper.is-customized {
  /* Set padding to ensure the height is 32px */
  padding: 6px 12px;
  background: linear-gradient(90deg, #d1c8fd, #e3eeff);
}

.el-popper.is-customized .el-popper__arrow::before {
  background: linear-gradient(45deg, #d1c8fd, #e3eeff); /*c0b7d8*/
  right: 0;
}

/*按钮 */
.el-button--primary { /*需要更改的按钮类型*/
  background: #a899ef !important;
  border-color: #FFF !important;
}
/*移入时按钮样式*/
.el-button--primary:hover {
  background: #d1c8fd !important;
  border-color: #FFF !important;
  color: #FFF !important;
}
</style>