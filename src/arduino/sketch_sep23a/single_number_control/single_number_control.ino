// ================== Pin Definitions ==================
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

// ================== Globals ==================
int lastPWM[4] = {0, 0, 0, 0};   // store last PWM for each motor

// ================== Setup ==================
void setup() {
  Serial.begin(9600);
  Serial.println("Motor control ready.");
  Serial.println("Send 4 numbers separated by commas (e.g., 200,-220,255,-180).");
  Serial.println("Each number: -255→-150=Backward, -149→149=Stop, 150→255=Forward");

  // Set PWM frequency to ~31kHz (faster, quieter motors)
  TCCR0B = (TCCR0B & 0b11111000) | 0x01;
  TCCR2B = (TCCR2B & 0b11111000) | 0x01;
}

// ================== Main Loop ==================
void loop() {
  // Check if a full line is available
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n'); // read until newline
    line.trim();
    if (line.length() > 0) {
      int values[4];
      if (parseInput(line, values)) {
        // Valid 4 numbers received, update lastPWM
        for (int i = 0; i < 4; i++) {
          lastPWM[i] = values[i];
        }
        Serial.print("Received PWM: ");
        for (int i = 0; i < 4; i++) {
          Serial.print(lastPWM[i]);
          if (i < 3) Serial.print(", ");
        }
        Serial.println();
      } else {
        Serial.println("Invalid input! Send 4 numbers separated by commas.");
      }
    }
  }

  // Continuously apply the last PWM values
  applyPWM();
}

// ================== Parse Input ==================
bool parseInput(String str, int* values) {
  int index = 0;
  while (index < 4) {
    int commaIndex = str.indexOf(',');
    String token;
    if (commaIndex == -1) {
      token = str;
    } else {
      token = str.substring(0, commaIndex);
    }
    token.trim();
    if (token.length() == 0) return false;
    values[index] = token.toInt();
    if (values[index] < -255) values[index] = -255;
    if (values[index] > 255)  values[index] = 255;
    index++;
    if (commaIndex == -1) break;
    str = str.substring(commaIndex + 1);
  }
  return index == 4; // success if 4 numbers were read
}

// ================== Apply stored PWM ==================
void applyPWM() {
  for (int i = 0; i < 4; i++) {
    int pwmValue = lastPWM[i];
    if (pwmValue == 0 || (pwmValue > -150 && pwmValue < 150)) {
      motor(i + 1, RELEASE, 0);
    } else if (pwmValue > 0) {
      motor(i + 1, FORWARD, pwmValue);
    } else {
      motor(i + 1, BACKWARD, abs(pwmValue));
    }
  }
}

// ================== Stop All Motors ==================
void stopAllMotors() {
  for (int i = 1; i <= 4; i++) {
    motor(i, RELEASE, 0);
  }
}

// ================== Set All Motors ==================
void setAllMotors(int command, int speed) {
  for (int i = 1; i <= 4; i++) {
    motor(i, command, speed);
  }
}

// ================== Motor Control ==================
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

// ================== Low-Level Control ==================
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
