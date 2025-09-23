#define MOTORLATCH   12
#define MOTORCLK     4
#define MOTORENABLE  7
#define MOTORDATA    8
#define MOTOR1_A  2
#define MOTOR1_B  3
#define MOTOR2_A  1
#define MOTOR2_B  4
#define MOTOR3_A  5
#define MOTOR3_B  7
#define MOTOR4_A  0
#define MOTOR4_B  6
#define MOTOR1_PWM 11
#define MOTOR2_PWM 3
#define MOTOR3_PWM 6
#define MOTOR4_PWM 5
#define FORWARD   1
#define BACKWARD  2
#define RELEASE   3
// ================== 初始化 ==================
void setup() {
  Serial.begin(9600);
  Serial.println("控制说明:");
  Serial.println("A: 电机1/2正转，电机3/4反转");
  Serial.println("S: 电机1/2反转，电机3/4正转");
  Serial.println("D: 电机1/3正转，电机2/4反转");
  Serial.println("F: 电机1/3反转，电机2/4正转");
  Serial.println("G: 电机1/2正转，电机3/4反转，占空比 50→255→保持→50 循环");
  Serial.println("H: 电机1/2:left_right_arrow:3/4交替正反转 + PWM呼吸 (50~255)");
  Serial.println("J: 电机1/2正转，电机3/4反转，可用 +/- 调整占空比");
  Serial.println("空格: 停止所有电机并退出循环模式");
  // ============= 修改 PWM 频率到 ~31kHz =============
  TCCR0B = (TCCR0B & 0b11111000) | 0x01;
  TCCR2B = (TCCR2B & 0b11111000) | 0x01;
  Serial.println("PWM 频率已调到 ~31kHz");
}
// ================== 主循环 ==================
void loop() {
  if (Serial.available()) {
    char key = Serial.read();
    key = toupper(key);
    switch (key) {
      case 'A':
        motor(1, FORWARD, 255);
        motor(2, FORWARD, 255);
        motor(3, BACKWARD, 255);
        motor(4, BACKWARD, 255);
        Serial.println("执行 A: 电机1/2正转，电机3/4反转");
        break;
      case 'S':
        motor(1, BACKWARD, 255);
        motor(2, BACKWARD, 255);
        motor(3, FORWARD, 255);
        motor(4, FORWARD, 255);
        Serial.println("执行 S: 电机1/2反转，电机3/4正转");
        break;
      case 'D':
        motor(1, FORWARD, 255);
        motor(3, FORWARD, 255);
        motor(2, BACKWARD, 255);
        motor(4, BACKWARD, 255);
        Serial.println("执行 D: 电机1/3正转，电机2/4反转");
        break;
      case 'F':
        motor(1, BACKWARD, 255);
        motor(3, BACKWARD, 255);
        motor(2, FORWARD, 255);
        motor(4, FORWARD, 255);
        Serial.println("执行 F: 电机1/3反转，电机2/4正转");
        break;
      // ========== G 模式 ==========
      case 'G':
        Serial.println("执行 G: 电机1/2正转，电机3/4反转，占空比 50→255→保持→50 循环");
        while (true) {
          // 前2秒：50 -> 255
          for (int step = 0; step <= 100; step++) {
            int pwm = map(step, 0, 100, 50, 255);
            motor(1, FORWARD, pwm);
            motor(2, FORWARD, pwm);
            motor(3, BACKWARD, pwm);
            motor(4, BACKWARD, pwm);
            delay(20);
          }
          // 保持1秒
          motor(1, FORWARD, 255);
          motor(2, FORWARD, 255);
          motor(3, BACKWARD, 255);
          motor(4, BACKWARD, 255);
          delay(1000);
          // 后2秒：255 -> 50
          for (int step = 0; step <= 100; step++) {
            int pwm = map(step, 0, 100, 255, 50);
            motor(1, FORWARD, pwm);
            motor(2, FORWARD, pwm);
            motor(3, BACKWARD, pwm);
            motor(4, BACKWARD, pwm);
            delay(20);
          }
          if (Serial.available()) {
            char key2 = Serial.read();
            if (key2 == ' ') {
              stopAllMotors();
              Serial.println(":warning: 退出 G 模式");
              break;
            }
          }
        }
        break;
      // ========== H 模式 ==========
      case 'H':
        Serial.println("执行 H: 电机1/2:left_right_arrow:3/4交替正反转 + PWM呼吸 (50~255)");
        while (true) {
          // 电机1/2正转, 电机3/4反转
          for (int pwm = 50; pwm <= 255; pwm++) {
            motor(1, FORWARD, pwm);
            motor(2, FORWARD, pwm);
            motor(3, BACKWARD, pwm);
            motor(4, BACKWARD, pwm);
            delay(10);
          }
          for (int pwm = 255; pwm >= 50; pwm--) {
            motor(1, FORWARD, pwm);
            motor(2, FORWARD, pwm);
            motor(3, BACKWARD, pwm);
            motor(4, BACKWARD, pwm);
            delay(10);
          }
          stopAllMotors();
          delay(1000);
          // 电机1/2反转, 电机3/4正转
          for (int pwm = 50; pwm <= 255; pwm++) {
            motor(1, BACKWARD, pwm);
            motor(2, BACKWARD, pwm);
            motor(3, FORWARD, pwm);
            motor(4, FORWARD, pwm);
            delay(10);
          }
          for (int pwm = 255; pwm >= 50; pwm--) {
            motor(1, BACKWARD, pwm);
            motor(2, BACKWARD, pwm);
            motor(3, FORWARD, pwm);
            motor(4, FORWARD, pwm);
            delay(10);
          }
          stopAllMotors();
          delay(1000);
          if (Serial.available()) {
            char key2 = Serial.read();
            if (key2 == ' ') {
              stopAllMotors();
              Serial.println(":warning: 退出 H 模式");
              break;
            }
          }
        }
        break;
      // ========== J 模式 ==========
      case 'J':
        Serial.println("执行 J: 电机1/2正转, 电机3/4反转, 用 +/- 调节占空比");
        {
          int pwmJ = 150;  // 初始占空比
          while (true) {
            motor(1, FORWARD, pwmJ);
            motor(2, FORWARD, pwmJ);
            motor(3, BACKWARD, pwmJ);
            motor(4, BACKWARD, pwmJ);
            if (Serial.available()) {
              char key2 = Serial.read();
              if (key2 == '+') {
                pwmJ = min(pwmJ + 10, 255);
                Serial.print("PWM 增加到: "); Serial.println(pwmJ);
              } else if (key2 == '-') {
                pwmJ = max(pwmJ - 10, 50);
                Serial.print("PWM 减少到: "); Serial.println(pwmJ);
              } else if (key2 == ' ') {
                stopAllMotors();
                Serial.println(":warning: 退出 J 模式");
                break;
              }
            }
            delay(50);
          }
        }
        break;
      case ' ':  // 空格停止
        stopAllMotors();
        Serial.println(":warning: 所有电机停止");
        break;
    }
  }
}
// ================== 停止所有电机 ==================
void stopAllMotors() {
  for (int i = 1; i <= 4; i++) {
    motor(i, RELEASE, 0);
  }
}
// ================== 电机控制函数 ==================
void motor(int nMotor, int command, int speed) {
  int motorA, motorB, motorPWM;
  switch (nMotor) {
    case 1: motorA = MOTOR1_A; motorB = MOTOR1_B; motorPWM = MOTOR1_PWM; break;
    case 2: motorA = MOTOR2_A; motorB = MOTOR2_B; motorPWM = MOTOR2_PWM; break;
    case 3: motorA = MOTOR3_A; motorB = MOTOR3_B; motorPWM = MOTOR3_PWM; break;
    case 4: motorA = MOTOR4_A; motorB = MOTOR4_B; motorPWM = MOTOR4_PWM; break;
    default: return;
  }
  switch (command) {
    case FORWARD:
      motor_output(motorA, HIGH, speed, motorPWM);
      motor_output(motorB, LOW, -1, motorPWM);
      break;
    case BACKWARD:
      motor_output(motorA, LOW, speed, motorPWM);
      motor_output(motorB, HIGH, -1, motorPWM);
      break;
    case RELEASE:
      motor_output(motorA, LOW, 0, motorPWM);
      motor_output(motorB, LOW, -1, motorPWM);
      break;
  }
}
void motor_output(int output, int high_low, int speed, int motorPWM) {
  shiftWrite(output, high_low);
  if (speed >= 0 && speed <= 255) {
    analogWrite(motorPWM, speed);
  }
}
void shiftWrite(int output, int high_low) {
  static int latch_copy;
  static bool init = false;
  if (!init) {
    pinMode(MOTORLATCH, OUTPUT);
    pinMode(MOTORENABLE, OUTPUT);
    pinMode(MOTORDATA, OUTPUT);
    pinMode(MOTORCLK, OUTPUT);
    digitalWrite(MOTORDATA, LOW);
    digitalWrite(MOTORLATCH, LOW);
    digitalWrite(MOTORCLK, LOW);
    digitalWrite(MOTORENABLE, LOW);
    latch_copy = 0;
    init = true;
  }
  bitWrite(latch_copy, output, high_low);
  shiftOut(MOTORDATA, MOTORCLK, MSBFIRST, latch_copy);
  digitalWrite(MOTORLATCH, HIGH);
  digitalWrite(MOTORLATCH, LOW);
}
