/**
 * This class provides the endpoint for the entity recognition task.
 */
package edu.ufl.cloudlang.rest;

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

@Path("/entities")
public class NamedEntityRecognizer {
	@GET
	@Produces("application/json")
	public Response parseText() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("testentitykey", "testentityvalue");
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
	
	@Path("{text}")
	@GET
	@Produces("application/json")
	public Response parseTextWithInput(@PathParam("text") String text) throws JSONException {
		JSONObject json =  new JSONObject();
		json.put("input text", text);
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
}
