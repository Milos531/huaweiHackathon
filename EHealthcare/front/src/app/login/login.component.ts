import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { User } from '../model/user';
import { UsercloudService } from '../usercloud.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  constructor(private userService: UsercloudService, private router: Router) { }

  ngOnInit(): void {
  }

  username: string;
  password: string;
  message: string;

  login(){
    this.userService.login(this.username, this.password).subscribe((userToken: User)=>{
      if(userToken!=null){
        localStorage.setItem('user',JSON.stringify(userToken))
        if(userToken.type=="patient"){
		      
          this.router.navigate(['patient']);
        }
        if (userToken.type=="doctor"){
          this.router.navigate(['doctor']);
        }
        else(
          alert("Error user!")
        )
      }
      else{
        this.message="Error"
      }
    })
    
  }
}
