//ADC 5 IS THE RESISTANCE
//ADC 6 IS THE POTENTIOMETER
void setup() {
  // put your setup code here, to run once:
Serial.begin(2400); // ADC6 left adjusted measurement //0b01 para 0-5V 0b11 para 0-1.1V
ADCSRA=0b11101010; //enable, start conversion, auto trigger,, adc interrupt
DDRD=0b1100000;
// we set the PWM to 7.8khz to control the motor
TCCR0A=0b10110011;
TCCR0B=0b00000010;
OCR0A=0;
OCR0B=10;

}

volatile uint16_t adc=0,pot=0;
volatile uint32_t current=0;
volatile uint8_t cont=0;
ISR(ADC_vect){
  //after 255 current measurement we obtain the voltage at the potentiometer (to control the speed).
 if(cont==0){
ADCSRA=0; // We disable de ADC and change teh configuration to read the potentiometer at channel 6
ADMUX= 0b11100110; // ADC6 left adjusted measuremen //0b01 para 0-5V 0b11 para 0-1.1V
ADCSRA=0b11010010; //enable, start conversion, 0,0,010 4 clock division 
while(ADCSRA&0x40)
    ;
pot=(63*pot+ADCH)>>6;
// we restore teh configuration of the ADC to measure teh current
ADMUX= 0b11100101; // ADC6 left adjusted measurement //0b01 para 0-5V 0b11 para 0-1.1V
cont++;
ADCSRA=0b11111010; //enable, start conversion, auto trigger,, adc interru{pt
}
else{ 
 adc=(31*adc+ADCH)>>5;
 cont++;}
}


void loop() {
  // put your main code here, to run repeatedly:
  current=(31*current+32*adc+400)>>5;
  OCR0A=pot;
  OCR0B=pot+7;
  if((cont&0xf)==0)
  {char cadena[256];
  sprintf(cadena,"%4d Current: %6ldma Duty:%5d\n",adc,current,pot);
  Serial.write(cadena);}

}
