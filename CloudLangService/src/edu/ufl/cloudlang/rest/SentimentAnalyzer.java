/**
 * This class provides the endpoint for the sentiment analysis task.
 */
package edu.ufl.cloudlang.rest;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Response;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * @author Sayak Biswas
 *
 */

@Path("/sentiment")
public class SentimentAnalyzer {
	@GET
	@Produces("application/json")
	public Response parseText() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("testsentimentkey", "testsentimentvalue");
		
		return Response.status(200).entity(json.toString()).build();
	}
	
	@Path("{text}")
	@GET
	@Produces("application/json")
	public Response parseTextWithInput(@PathParam("text") String text) throws JSONException {
		String command = "THEANO_FLAGS=\"floatX=float32\" python lstm.py '" + text + "'";
		Process process = null;
		String log = null;
		String sentimentResult = null;
		BufferedReader stdInput = null;
		try {
			System.out.println("Will execute command " + command);
			process = Runtime.getRuntime().exec(command);
			process.waitFor();
			if(process != null) {
				stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
				System.out.println("Process logs :: ");
				while ((log = stdInput.readLine()) != null) {
					System.out.println(log);
				}
			}
		} catch (IOException ie) {
			ie.printStackTrace();
		} catch (InterruptedException ie) {
			ie.printStackTrace();
		}
		sentimentResult = log;
		JSONObject json =  new JSONObject();
		json.put("inputText", text);
		json.put("sentimentResult", sentimentResult);
		
		return Response.status(200).entity(json.toString()).build();
	}
}
