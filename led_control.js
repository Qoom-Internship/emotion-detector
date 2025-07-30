const { execSync } = require('child_process');

const gpioPins = [17, 18, 27, 22, 23];

gpioPins.forEach(pin => {
  execSync(`gpioset 0 ${pin}=1`);
});
console.log('LED 5개 ON');

setTimeout(() => {
  gpioPins.forEach(pin => {
    execSync(`gpioset 0 ${pin}=0`);
  });
  console.log('LED 5개 OFF');
}, 3000);