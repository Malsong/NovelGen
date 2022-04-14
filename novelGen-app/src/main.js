import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'
import axios from "axios"
import qs from "qs"
import store from './store/store.js'
//引入全部组件及样式
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css';
import * as Elicons from '@element-plus/icons-vue' // 全局引用element-icon


const app = createApp(App)

// 使用暴露出来的router
app.use(router);
app.use(ElementPlus);
app.use(store);
// 注册Icons 全局组件
Object.keys(Elicons).forEach(key => {
    app.component(key, Elicons[key])
  })
// 挂载根组件App.vue
app.mount('#app');

/* axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded'; */
app.config.globalProperties.$http=axios;
app.config.globalProperties.$qs = qs;
/* axios.defaults.baseURL = '/api'  //关键代码 */
