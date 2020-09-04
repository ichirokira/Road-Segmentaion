<template>
  <div id='segment' class="segmentation">
      <h1 class="instagram">Road Segmentation</h1>
      <br>
      <center>
        <div class='frame'>
          <video width="960" height="540" controls>
            <source src="../assets/test3_output.mp4" type='video/mp4'>
          </video>
          <div v-if="this.status == 'done'">
          <span class='gradient-fill'>Frame per second: </span>
          <span>{{ f }}</span>
          <br>
           <span class='gradient-fill'>Total time to run the video: </span>
          <span>{{ time }}</span>
        </div> 
        </div>
        
        
      </center>

      
  </div>
</template>
<script>
import axios from 'axios';

export default {
  name: 'segmentation',
  data() {
    return {
      status: "started",
      f: 0,
      time: 0
    };
  },
  mounted() {
    this.status = "loading";
    axios
      .get("http://0.0.0.0:8084/output")
      .then((response) =>{
        console.log(response.data.fps)
        this.f = response.data.fps;
        console.log(this.f)
        this.time = response.data.total_time;
        this.status = 'done'
      })
      .catch((error) => {
        console.error(error);
      });
  },
  //console.log(fps)
};
//console.log(this.fps)

</script>
<style scoped>
.segmentation {
  text-align: left;
  font-family: Inter, Inter UI, Inter-UI, SF Pro Display, SF UI Text,
    Helvetica Neue, Helvetica, Arial, sans-serif;
  font-weight: 400;
  letter-spacing: +0.37px;
  color: rgb(175, 175, 175);
}
img {
    border-radius: 5%;
}
.frame {
  width: 960px;
  height: 540px;
  border: 5px solid #ae41a7;
  border-radius: 5%;
  background: #eee;
  margin: center;
  padding: 15px 10px;
}
.form-control:focus {
  border-color: #ae41a7 !important;
  box-shadow: 0 0 5px #ae41a7 !important;
}
.instagram{
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background: #f09433;
  background: -moz-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  background: -webkit-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #806878 100%
  );
  background: linear-gradient(45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  filter: progid:DXImageTransform.Microsoft.gradient( startColorstr=#f09433,
    endColorstr=#bc1888,
    GradientType=1
  );
}
.header {
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}
.gradient-fill {
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}
.gradient-fill.background {
  background-size: 250% auto;
  border: medium none currentcolor;
  border-image: none 100% 1 0 stretch;
  transition-delay: 0s, 0s, 0s, 0s, 0s, 0s;
  transition-duration: 0.5s, 0.2s, 0.2s, 0.2s, 0.2s, 0.2s;
  transition-property: background-position, transform, box-shadow, border,
    transform, box-shadow;
  transition-timing-function: ease-out, ease, ease, ease, ease, ease;
  color: white;
  font-weight: bold;
  border-radius: 3px;
}
span.gradient-fill {
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 20px;
  font-weight: 700;
  line-height: 2.5;
}
</style>
 