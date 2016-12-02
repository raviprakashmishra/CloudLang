/**
 * This class provides the endpoint for the sentiment analysis task.
 */
package edu.ufl.cloudlang.rest;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.core.Response;

import org.glassfish.jersey.client.ClientConfig;
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
		Client client = ClientBuilder.newClient(new ClientConfig());
		String sentimentResult = client.target("http://localhost:5000").path("sentiment").path(text).request().get(String.class);
		System.out.println("sentimentResult " + sentimentResult);
		
		JSONObject json =  new JSONObject();
		json.put("inputText", text);
		json.put("sentimentResult", sentimentResult);
		
		return Response.status(200).entity(json.toString()).build();
	}
}
