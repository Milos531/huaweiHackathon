import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class UsercloudService {

  constructor(private http: HttpClient) { }

  uri = 'http://localhost:8000/ehealthcare'

  login(username, password){
    const data = {
      username: username,
      password: password
    }

    return this.http.post(`${this.uri}/login`, data)
  }

  register(username, email, password, type, surename, forename){
    const data = {
      forename: forename,
      surename: surename,
      username: username,
      password: password,
      type: type,
      email: email, 
    }

    return this.http.post(`${this.uri}/register`, data)
  }
}
