import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { UsercloudService } from '../usercloud.service';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  constructor(private userService: UsercloudService, private router: Router) { }

  ngOnInit(): void {
  }

  firstname: string
  lastname: string
  username: string
  password: string
  type: string
  email: string

  message: string;

  register(){
    if (!this.firstname  || !this.email || !this.lastname || !this.password || !this.username || !this.type)
      this.message = "All fields are required!"
      
    else{
      this.userService.register(this.username, this.email, this.type, this.password, this.lastname,this.firstname ).subscribe((msg : {"message": string})=>{
        if(!msg){
          this.message = 'No response'
        }
        else{
          alert(msg.message)
          this.router.navigate(['login'])
        }
      });
    }
  }
}
