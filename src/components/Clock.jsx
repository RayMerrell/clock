import React, { useEffect } from "react";
import { useRef } from "react";

/** @type {HTMLCanvasElement} */
/** @type {WebGL2RenderingContext} */
const vertexShaderSource = `#version 300 es
precision mediump float;
layout (location=0) in vec2 uPosition;
void main()
  {
    gl_Position = vec4(uPosition, 0.0, 1.0);
  }
`;
const fragmentShaderSource = `#version 300 es
precision mediump float;
uniform vec2 uResolution;
uniform vec2 uMouse;
uniform int iAnimationFrame;
uniform int iHours;
uniform int iMinutes;
uniform int iSeconds;
uniform int iMillis;
uniform sampler2D uTexture;
const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 10.0;
const float PRECISION = 0.00001;
const float PI = 3.1415926535897932;
const float TAU = 6.283185307179586;

out vec4 fragColour;

struct Material{
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  float shininess;
};
struct Surface{
  int id;
  float sdf;
  Material material;
};
Surface sdfScene(vec3 position);
Material gold() {
  vec3 aCol = 0.5 * vec3(0.7, 0.5, 0);
  vec3 dCol = 0.6 * vec3(0.7, 0.7, 0);
  vec3 sCol = 0.85 * vec3(0.8, 1, 0.8);
  float a = 9.0;
  return Material(aCol, dCol, sCol, a);
}
Material silver() {
  vec3 aCol = 0.4 * vec3(0.8);
  vec3 dCol = 0.5 * vec3(0.7);
  vec3 sCol = 0.9 * vec3(1, 1, 1);
  float a = 10.0;
  return Material(aCol, dCol, sCol, a);
}
Material checkerboard(vec3 p) {
  vec3 aCol = vec3(1. + 0.7*mod(floor(p.x) + floor(p.z), 2.0)) * 0.3;
  vec3 dCol = vec3(p.x*0.95, 0.0, p.z*0.9);
  vec3 sCol = vec3(0);
  float a = 0.2;

  return Material(aCol, dCol, sCol, a);
}
// Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

// Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

// Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

// Identity matrix.
mat3 identity() {
    return mat3(
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(0, 0, 1)
    );
}

Surface sdfCone(vec3 position, vec3 a, vec3 b, float ra, float rb, Material mat, int id)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(position-a,position-a);
    float paba = dot(position-a,b-a)/baba;

    float x = sqrt( papa - paba*paba*baba );

    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;

    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );

    float cbx = x-ra - f*rba;
    float cby = paba - f;
    
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    Surface ret = Surface(id, 0.0, mat);
    ret.sdf =  s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
    return ret;
}

Surface sdfRoundedCylinder( vec3 position, float ra, float rb, float h, Material mat, int id ){  
  vec2 d = vec2( length(position.xz)-2.0*ra+rb, abs(position.y) - h );
  Surface ret = Surface(id, 0.0, mat);
  ret.sdf= min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
  return ret;
}
Surface sdfCappedCylinder( vec3 p, float h, float r , Material mat, int id){
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  Surface ret = Surface(id, 0.0, mat);
  ret.sdf= min(max(d.x,d.y),0.0) + length(max(d,0.0));
  return ret;
}
Surface sdfSphere(vec3 position, float radius, vec3 offset, Material mat, int id){
  float dist =length(position - offset) - radius;
  return Surface(id, dist, mat);
}

Surface sdfBackground(vec3 position, Material mat, int id){
  float dist = position.y + 0.0;
  return Surface(id, dist, mat);
}

Surface sdfBox(vec3 position, vec3 boundries,vec3 location, Material mat, int id){
  position = position - location;
  vec3 q = abs(position) - boundries;
  float dist = length(max(q,0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  return Surface(id, dist, mat);
}

Surface sdfRoundBox(vec3 position, vec3 boundries, float radius, Material mat, int id){
  vec3 q = abs(position) - boundries;
  float dist = length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - radius;
  return Surface(id, dist, mat);
}

Surface rayMarch(vec3 origin, vec3 direction){
  float depth = MIN_DIST;
  Surface closestObject;

  for (int i=0; i<MAX_MARCHING_STEPS; i++){
    vec3 position = origin + depth * direction;
    closestObject = sdfScene(position);
    depth += closestObject.sdf;
    if (closestObject.sdf < PRECISION || depth > MAX_DIST) break;
  }
  if (depth > MAX_DIST) depth = -1.0;
  closestObject.sdf = depth;
  return closestObject;
}

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0005, -0.0005); // epsilon  
  return normalize(
    e.xyy * sdfScene(p + e.xyy).sdf +
    e.yyx * sdfScene(p + e.yyx).sdf +
    e.yxy * sdfScene(p + e.yxy).sdf +
    e.xxx * sdfScene(p + e.xxx).sdf);
 
}
Surface opUnion(Surface s1, Surface s2){
  if (s1.sdf < s2.sdf)return s1;
  return s2;
}

Surface opSmoothUnion( Surface d1, Surface d2, float k ) {
  float h = clamp( 0.5 + 0.5*(d2.sdf-d1.sdf)/k, 0.0, 1.0 );
  Surface ret = Surface(d1.id, 0.0, d1.material);
  ret.sdf= mix( d2.sdf, d1.sdf, h ) - k*h*(1.0-h); 
  return ret;
}

Surface opSmoothSubtraction( Surface d1, Surface d2, float k ) {
  float h = clamp( 0.5 - 0.5*(d2.sdf+d1.sdf)/k, 0.0, 1.0 );
  Surface ret = Surface(d1.id, 0.0, d1.material);
  ret.sdf = mix( d2.sdf, -d1.sdf, h ) + k*h*(1.0-h); 
  return ret;
}

float opSmoothIntersection( Surface d1, Surface d2, float k ) {
  float h = clamp( 0.5 - 0.5*(d2.sdf-d1.sdf)/k, 0.0, 1.0 );
  return mix( d2.sdf, d1.sdf, h ) + k*h*(1.0-h); }

vec3 phong (vec3 lightDirection, vec3 normal, vec3 rayDirection, Material mat){
  //ambient
  vec3 ambient = mat.ambient;
  //diffuse
  float dotLN = clamp(dot(normal, lightDirection), 0.0, 1.0);
  vec3 diffuse = mat.diffuse * vec3(dotLN);
  //specular
  float dotRV = clamp(dot(reflect(lightDirection, normal), - rayDirection), 0.0, 1.0);
  vec3 specular = mat.specular * pow(dotRV, mat.shininess);
  return ambient + diffuse + specular;
}
mat3 rotateSeconds(){
  float rads = (float(iSeconds)/60.0)* TAU + (float(iMillis)/1000.0/60.0) * TAU;
  float c = cos(rads);
  float s = sin(rads);
  return mat3(
    vec3(c, 0, s),
    vec3(0, 1, 0),
    vec3(-s, 0, c)
);
}
mat3 rotateMinutes(){
  float rads = (float(iMinutes)/60.0)* TAU + (float(iSeconds)/3600.0)* TAU + (float(iMillis)/1000.0/3600.0)*TAU;
  float c = cos(rads);
  float s = sin(rads);
  return mat3(
    vec3(c, 0, s),
    vec3(0, 1, 0),
    vec3(-s, 0, c)
);
}
mat3 rotateHours(){
  float rads = (float(iHours)/12.0)* TAU + (float(iMinutes)/60.0/12.0) * TAU +(float(iSeconds)/3600.0/12.0)* TAU;
  float c = cos(rads);
  float s = sin(rads);
  return mat3(
    vec3(c, 0, s),
    vec3(0, 1, 0),
    vec3(-s, 0, c)
);
}
Surface sdfSecondHand(vec3 position, Material mat, int id){
  vec3 offSet = vec3(0.0, 0.0704, -0.3);
  mat3 transform = rotateSeconds();
  Surface secondHand = sdfRoundBox(position* transform-offSet , vec3(0.005125, 0.0001, 0.5), 0.001, mat,id);
  return secondHand;
}
Surface sdfMinuteHand(vec3 position, Material mat, int id){
  vec3 offSet = vec3(0.0, 0.06, -0.3);
  mat3 transform = rotateMinutes();
  Surface secondHand = sdfRoundBox(position* transform-offSet , vec3(0.025125, 0.0001, 0.55), 0.001, mat,id);
  return secondHand;
}
Surface sdfHourHand(vec3 position, Material mat, int id){
  vec3 offSet = vec3(0.0, 0.05, -0.3);
  mat3 transform = rotateHours();
  Surface secondHand = sdfRoundBox(position* transform-offSet , vec3(0.03, 0.0001, 0.35), 0.001, mat,id);
  return secondHand;
}
Surface sdfScene(vec3 position){
  //Surface floor = sdfBackground(position,checkerboard(position), 1);
  Surface bezel = sdfCone(position, vec3(0,0.090,0), vec3(0,0.12,0), 1.20, 1.0, gold(), 2);
  //bezel.sdf = smoothstep (0.01, 0.99, bezel.sdf);
  Surface outerBody = sdfCappedCylinder(position, 0.1, 1.225, gold(),3);
  //Surface innerMask = sdfRoundedCylinder(position-vec3(0.0,-0.1,0.0), 0.05, 1.1, 0.0015,gold(), 4);
  vec3 q = position-vec3(0.0,0.21,0.0);
  //base grooves
  float offset = (pow(sin(position.z * 550.0)*0.335, 7.0)/2.5);
  Surface innerMask = sdfCappedCylinder(q - offset, 0.2, 1.0, gold(),4);
  q = (position * rotateZ(PI/2.0)) - vec3(0.0,1.2 + 0.03,0.0);
  Surface winder = sdfRoundedCylinder(q, 0.045, 0.01, 0.03,gold(), 5);
  Surface body = opSmoothUnion(outerBody, bezel, 0.02);
  body = opSmoothSubtraction( innerMask, body,0.025);  
  Surface centrePin = sdfRoundedCylinder(position, 0.00525, 0.005 , 0.075, gold(), 6);
  Surface secondHand = sdfSecondHand (position, silver(), 7);  
  Surface minuteHand = sdfMinuteHand (position, silver(), 8);
  Surface hourHand = sdfHourHand (position, silver(), 9);
  
  body = opUnion(body, centrePin);
  body = opUnion(body, secondHand);  
  body = opUnion(body, minuteHand);
  body = opUnion(body, hourHand);
  body = opUnion(body, winder);
  
  //Surface co = opUnion(body, floor);

  return body;
}
void main(){
    vec4 colour = vec4(0);  //output colour
    vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y; //get our virtual world coords (-0.5 => +0.5)
    vec2 mouse = (uMouse.xy -0.5 * uResolution.xy)/ uResolution.y;  //mouse to virtual world coords
    float cameraAngle = 3.0 * abs(mouse.y);
    vec3 cameraTarget = vec3(0.0, 0.0, 0.0);
    
    //vec3 rayOrigin = vec3(1.5*sin(cameraAngle), 0.25, 5.0*cos(cameraAngle));
    //vec3 rayOrigin = vec3(0.0, 5.0, 0.00001);
    vec3 rayOrigin = vec3(0.0, 2.35*(cameraAngle), 0.0001);
    vec3 ww = normalize(cameraTarget - rayOrigin);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    vec3 rayDirection = normalize(uv.x*uu + uv.y * vv + 1.5*ww);


    Surface closestObject =rayMarch(rayOrigin, rayDirection);
    if (closestObject.sdf < 0.0){
      //colour += vec3(0.4,0.75,1.0) - 1.5 * rayDirection.y; //sky gradient
      //colour = mix(colour, vec3(0.7,0.75,0.8), exp(-10.0*rayDirection.y));//horizon fog

    }else{
      vec3 point = rayOrigin + rayDirection * closestObject.sdf; //point of contact
      vec3 normal = calcNormal(point);
      //sunlight
      vec3 sunPosition = vec3(-8.0 * sin(float(iAnimationFrame)/500.0), 10.0, -5.0 * cos(float(iAnimationFrame)/500.0));
      vec3 sunDirection = normalize (sunPosition - point);
      float sunIntensity = 0.9;
      //sky      
      vec3 skyPosition = vec3(0.0, 10.0,0.0);
      vec3 skyDirection = normalize(skyPosition-point);
      float skyIntensity = 0.25;
 
      vec3 lightPosition = vec3(2,30,7);
      vec3 lightDirection = normalize(lightPosition - point);
      float lightIntensity = 0.25;

      float bounceDiffuse = clamp(0.5 + 0.5 * dot(normal, vec3(0,-1,0)), 0.0,1.0);

      vec3 sunColour = vec3 (0.875, 0.875, 0.5) * vec3(sunIntensity * phong(sunDirection, normal, rayDirection, closestObject.material));  
      vec3 skyColour =  vec3 (0.125,0.125, 0.5) * vec3(skyIntensity * phong(skyDirection, normal, rayDirection, closestObject.material));

       //shadows
      vec3 newRayOrigin = point + (normal * PRECISION * 1.25);
      float shadowRayLength = rayMarch(newRayOrigin, sunDirection).sdf; // cast shadow ray to the light source
      if (shadowRayLength > -1.0 && shadowRayLength < length(sunPosition - newRayOrigin)) sunColour *= 0.5; // shadow

      newRayOrigin = point + (normal * PRECISION * 1.25);     
      shadowRayLength = rayMarch(newRayOrigin, skyDirection).sdf; // cast shadow ray to the light source
      if (shadowRayLength > -1.0 && shadowRayLength < length(skyPosition - newRayOrigin)) skyColour *= 0.5; // shadow
      //colour = vec4(mix(sunColour, skyColour, bounceDiffuse),1.0);
      colour = vec4((sunColour + skyColour) + (bounceDiffuse*0.01), 1.0);
      //colour = vec4(sunColour, 1.0);
    }


    //colour = vec4(pow(colour.rgb, vec3(0.7545)), colour.a);
    fragColour = colour;
}
`;

const initWebGL = (canv) => {
  const canvas = canv.current;
  const gc = canvas.getContext("webgl2");
  gc.htmlCanvasElement = canvas; //make canvas element a custom property
  console.log("In clockCode");
  const program = gc.createProgram();

  const vertexShader = gc.createShader(gc.VERTEX_SHADER);
  gc.shaderSource(vertexShader, vertexShaderSource);
  gc.compileShader(vertexShader);
  gc.attachShader(program, vertexShader);

  const fragmentShader = gc.createShader(gc.FRAGMENT_SHADER);
  gc.shaderSource(fragmentShader, fragmentShaderSource);
  gc.compileShader(fragmentShader);
  gc.attachShader(program, fragmentShader);

  gc.linkProgram(program);
  if (!gc.getProgramParameter(program, gc.LINK_STATUS)) {
    console.log(gc.getShaderInfoLog(vertexShader));
    console.log(gc.getShaderInfoLog(fragmentShader));
  }

  gc.useProgram(program);

  const uPositionLoc = gc.getUniformLocation(program, "uPosition");
  const uResolutionLoc = gc.getUniformLocation(program, "uResolution");
  const uMouseLoc = gc.getUniformLocation(program, "uMouse");
  const uAnimationFrameLoc = gc.getUniformLocation(program, "iAnimationFrame");
  const uHoursLoc = gc.getUniformLocation(program, "iHours");
  const uMinutesLoc = gc.getUniformLocation(program, "iMinutes");
  const uSecondsLoc = gc.getUniformLocation(program, "iSeconds");
  const uMillisLoc = gc.getUniformLocation(program, "iMillis");
  //add uniformLocations as custom properties on the graphics Context
  gc.uResolutionLocation = uResolutionLoc;
  gc.uMouseLocation = uMouseLoc;
  gc.uAnimationFrameLocation = uAnimationFrameLoc;
  gc.uHoursLocation = uHoursLoc;
  gc.uMinutesLocation = uMinutesLoc;
  gc.uSecondsLocation = uSecondsLoc;
  gc.uMillisLocation = uMillisLoc;

  gc.enableVertexAttribArray(uPositionLoc);
  const positionBuffer = gc.createBuffer();
  gc.bindBuffer(gc.ARRAY_BUFFER, positionBuffer);
  gc.bufferData(
    gc.ARRAY_BUFFER,
    new Float32Array([
      -1,
      -1, // first triangle
      1,
      -1,
      -1,
      1,
      -1,
      1, // second triangle
      1,
      -1,
      1,
      1,
    ]),
    gc.STATIC_DRAW
  );

  gc.vertexAttribPointer(
    uPositionLoc,
    2, // 2 components per iteration
    gc.FLOAT, // the data is 32bit floats
    false, // don't normalize the data
    0, // 0 = move forward size * sizeof(type) each iteration to get the next position
    0 // start at the beginning of the buffer
  );
  return gc;
};

//mouse movement
let globalMouseX = 0;
let globalMouseY = 0;
const setMousePos = (e) => {
  globalMouseX = e.clientX;
  globalMouseY = e.clientY;
};

const drawCanvas = (gc, mx, my, t) => {
  my.current.value = "";
  const today = new Date();
  gc.htmlCanvasElement.width = gc.htmlCanvasElement.offsetWidth;
  gc.htmlCanvasElement.height = gc.htmlCanvasElement.offsetHeight;
  const rect = gc.htmlCanvasElement.getBoundingClientRect();
  const mousex = globalMouseX - rect.left;
  const mousey = rect.height - (globalMouseY - rect.top); // bottom is 0 in WebGL
  mx.current.value = today.getHours();
  //my.current.value = mousey;
  gc.uniform2f(gc.uMouseLocation, mousex, mousey);
  gc.viewport(0, 0, gc.htmlCanvasElement.width, gc.htmlCanvasElement.height);
  gc.uniform2f(
    gc.uResolutionLocation,
    gc.htmlCanvasElement.width,
    gc.htmlCanvasElement.height
  );
  gc.uniform1i(gc.uAnimationFrameLocation, t);
  gc.uniform1i(gc.uHoursLocation, today.getHours());
  gc.uniform1i(gc.uMinutesLocation, today.getMinutes());
  gc.uniform1i(gc.uSecondsLocation, today.getSeconds());
  gc.uniform1i(gc.uMillisLocation, today.getMilliseconds());
  my.current.value = t;
  gc.drawArrays(gc.TRIANGLES, 0, 6);
};

const Clock = () => {
  const canvas = useRef(null);
  const my = useRef(null);
  const mx = useRef(null);
  let t = 0;
  const animate = (gc) => {
    t++;
    drawCanvas(gc, mx, my, t);
    requestAnimationFrame(() => animate(gc));
  };
  useEffect(() => {
    const gc = initWebGL(canvas);
    canvas.current.addEventListener("mousemove", setMousePos);
    requestAnimationFrame(() => animate(gc));
  }, []);

  return (
    <>
      X<input type="text" ref={mx}></input>Y<input type="text" ref={my}></input>
      <canvas ref={canvas} className="mainCanvas"></canvas>
    </>
  );
};

export default Clock;
